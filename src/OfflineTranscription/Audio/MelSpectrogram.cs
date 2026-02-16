using System.Numerics;

namespace OfflineTranscription.Audio;

/// <summary>
/// Whisper-compatible mel spectrogram computation.
/// Matches python_simple_implementation.py: STFT with 400-sample Hann window,
/// 160-sample hop, 128 Slaney mel filterbank, log-mel scaling.
/// </summary>
internal static class MelSpectrogram
{
    private const int SampleRate = 16000;
    private const int NumMelBins = 128;
    private const int HopLength = 160;
    private const int WindowSize = 400; // n_fft
    private const int FftSize = WindowSize; // Same as n_fft for this model
    private const int NumFreqBins = FftSize / 2 + 1; // 201

    // Lazily computed mel filterbank [NumFreqBins, NumMelBins] = [201, 128]
    private static readonly float[] MelFilters = ComputeMelFilters();

    /// <summary>
    /// Compute mel spectrogram from 16kHz float audio.
    /// Returns float[128 * numFrames] in row-major order (128 mel bins x numFrames).
    /// </summary>
    public static (float[] data, int melBins, int numFrames) Compute(float[] audio)
    {
        // Number of STFT frames (matching torch.stft default: centered, no padding needed
        // since python_simple_implementation.py doesn't pad)
        int numFrames = audio.Length / HopLength;
        if (numFrames <= 0)
            return (Array.Empty<float>(), NumMelBins, 0);

        // Pre-compute Hann window
        var window = new float[WindowSize];
        for (int i = 0; i < WindowSize; i++)
            window[i] = 0.5f * (1.0f - MathF.Cos(2.0f * MathF.PI * i / WindowSize));

        // STFT: compute magnitude squared for each frame
        // magnitudes[frame, freq] for freq in [0, NumFreqBins)
        // Python: stft[..., :-1].abs() ** 2 drops the last frame
        int stftFrames = numFrames + 1; // torch.stft produces this many, then we drop last
        var magnitudes = new float[numFrames * NumFreqBins];

        // FFT buffer (power of 2 >= WindowSize; 512 for WindowSize=400)
        int fftN = 1;
        while (fftN < WindowSize) fftN <<= 1;
        var fftBuffer = new Complex[fftN];

        for (int frame = 0; frame < numFrames; frame++)
        {
            int center = frame * HopLength;

            // Fill FFT buffer with windowed samples
            Array.Clear(fftBuffer, 0, fftN);
            for (int j = 0; j < WindowSize; j++)
            {
                int sampleIdx = center + j - WindowSize / 2;
                float sample = (sampleIdx >= 0 && sampleIdx < audio.Length) ? audio[sampleIdx] : 0f;
                fftBuffer[j] = new Complex(sample * window[j], 0);
            }

            // In-place FFT
            Fft(fftBuffer, false);

            // Magnitude squared for first NumFreqBins bins
            int baseIdx = frame * NumFreqBins;
            for (int k = 0; k < NumFreqBins; k++)
            {
                double re = fftBuffer[k].Real;
                double im = fftBuffer[k].Imaginary;
                magnitudes[baseIdx + k] = (float)(re * re + im * im);
            }
        }

        // Apply mel filterbank: mel_spec[mel, frame] = sum over freq of filters[freq, mel] * magnitudes[frame, freq]
        var melSpec = new float[NumMelBins * numFrames];
        for (int frame = 0; frame < numFrames; frame++)
        {
            int magBase = frame * NumFreqBins;
            for (int mel = 0; mel < NumMelBins; mel++)
            {
                float sum = 0;
                for (int freq = 0; freq < NumFreqBins; freq++)
                    sum += MelFilters[freq * NumMelBins + mel] * magnitudes[magBase + freq];
                melSpec[mel * numFrames + frame] = sum;
            }
        }

        // Log-mel scaling (matching Python):
        // log_spec = log10(max(mel_spec, 1e-10))
        // log_spec = max(log_spec, log_spec.max() - 8.0)
        // log_spec = (log_spec + 4.0) / 4.0
        float globalMax = float.NegativeInfinity;
        for (int i = 0; i < melSpec.Length; i++)
        {
            melSpec[i] = MathF.Log10(MathF.Max(melSpec[i], 1e-10f));
            if (melSpec[i] > globalMax) globalMax = melSpec[i];
        }

        float clampMin = globalMax - 8.0f;
        for (int i = 0; i < melSpec.Length; i++)
        {
            melSpec[i] = MathF.Max(melSpec[i], clampMin);
            melSpec[i] = (melSpec[i] + 4.0f) / 4.0f;
        }

        return (melSpec, NumMelBins, numFrames);
    }

    /// <summary>
    /// Compute Slaney-style mel filterbank matching WhisperFeatureExtractor.
    /// Returns float[NumFreqBins * NumMelBins] in row-major [freq, mel] order.
    /// </summary>
    private static float[] ComputeMelFilters()
    {
        var fftFreqs = new float[NumFreqBins];
        for (int i = 0; i < NumFreqBins; i++)
            fftFreqs[i] = (float)i * SampleRate / 2.0f / (NumFreqBins - 1);

        float melMin = HertzToMel(0f);
        float melMax = HertzToMel(8000f);

        var melFreqs = new float[NumMelBins + 2];
        for (int i = 0; i < NumMelBins + 2; i++)
            melFreqs[i] = melMin + (melMax - melMin) * i / (NumMelBins + 1);

        var filterFreqs = new float[NumMelBins + 2];
        for (int i = 0; i < NumMelBins + 2; i++)
            filterFreqs[i] = MelToHertz(melFreqs[i]);

        var filterDiff = new float[NumMelBins + 1];
        for (int i = 0; i < NumMelBins + 1; i++)
            filterDiff[i] = filterFreqs[i + 1] - filterFreqs[i];

        var fb = new float[NumFreqBins * NumMelBins];
        for (int freq = 0; freq < NumFreqBins; freq++)
        {
            for (int mel = 0; mel < NumMelBins; mel++)
            {
                float downSlope = -(fftFreqs[freq] - filterFreqs[mel]) / filterDiff[mel];
                float upSlope = (fftFreqs[freq] - filterFreqs[mel + 2]) / filterDiff[mel + 1];
                float val = MathF.Max(0, MathF.Min(downSlope, upSlope));

                // Slaney normalization
                float enorm = 2.0f / (filterFreqs[mel + 2] - filterFreqs[mel]);
                fb[freq * NumMelBins + mel] = val * enorm;
            }
        }

        return fb;
    }

    private static float HertzToMel(float freq)
    {
        const float minLogHertz = 1000f;
        const float minLogMel = 15f;
        float logStep = 27f / MathF.Log(6.4f);
        float mel = 3f * freq / 200f;
        if (freq >= minLogHertz)
            mel = minLogMel + MathF.Log(freq / minLogHertz) * logStep;
        return mel;
    }

    private static float MelToHertz(float mel)
    {
        const float minLogHertz = 1000f;
        const float minLogMel = 15f;
        float logStep = MathF.Log(6.4f) / 27f;
        float freq = 200f * mel / 3f;
        if (mel >= minLogMel)
            freq = minLogHertz * MathF.Exp(logStep * (mel - minLogMel));
        return freq;
    }

    /// <summary>
    /// Cooley-Tukey radix-2 FFT (in-place). n must be a power of 2.
    /// </summary>
    private static void Fft(Complex[] buffer, bool inverse)
    {
        int n = buffer.Length;
        if (n <= 1) return;

        // Bit-reversal permutation
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            while ((j & bit) != 0) { j ^= bit; bit >>= 1; }
            j ^= bit;
            if (i < j) (buffer[i], buffer[j]) = (buffer[j], buffer[i]);
        }

        // Butterfly
        for (int len = 2; len <= n; len <<= 1)
        {
            double angle = (inverse ? 1 : -1) * 2.0 * Math.PI / len;
            var wn = new Complex(Math.Cos(angle), Math.Sin(angle));
            for (int i = 0; i < n; i += len)
            {
                var w = Complex.One;
                int half = len / 2;
                for (int j = 0; j < half; j++)
                {
                    var u = buffer[i + j];
                    var t = w * buffer[i + j + half];
                    buffer[i + j] = u + t;
                    buffer[i + j + half] = u - t;
                    w *= wn;
                }
            }
        }

        if (inverse)
        {
            for (int i = 0; i < n; i++)
                buffer[i] /= n;
        }
    }
}
