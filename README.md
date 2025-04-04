# Cassini SAR denoising 🛰️ 🌌

Project Members: Brian Li, Alan Yu, Arnold Su (listener), Kai-Feng Chen






Project contributed by [W. Hamish Mitchell](https://scholar.google.com/citations?user=TXmRX3UAAAAJ&hl=en) who is happy to provide more guidance.

## Project Description

### Cassini Mission and Titan 🪐

The Cassini mission to Saturn revealed one of its moons, **Titan**, as a remarkably Earth-like world. Despite its frigid surface temperature of -180°C, Titan exhibits stunning morphological features—dune fields, river networks, lakes, and seas—that bear striking similarities to those on Earth. This has made Titan one of the most intriguing targets for studying Earth-like processes on another world.

### Imaging Titan with SAR 📡

Due to Titan's thick atmosphere, the primary means of imaging its surface was through Cassini's **Synthetic Aperture Radar (SAR)**. SAR technology is instrumental in planetary observation because it can penetrate opaque atmospheres and operate independently of solar illumination. This capability allows SAR to function regardless of "optical visibility" conditions. As an active remote sensing technique, SAR sensors transmit microwave signals in the range direction—perpendicular to the sensor's flight path—and capture the returned echoes. The sensor records both the time delay of the echoes, which determines distance, and the relative phase of repeated echoes. Each pixel in the resulting image has a unique combination of range (distance from the sensor) and Doppler (frequency shift depending on the relative motion between the sensor and the imaged surface).

### Challenges with Speckle Noise 🔍

However, SAR images inherently suffer from **speckle noise**, which arises from the processing of radar returns. As demonstrated by [1], this speckle noise is multiplicative rather than additive. Specifically, for a SAR image, the relationship can be expressed as:

\[ |x(i,j)|² = |x₀(i,j)|²|n(i,j)|² \]

where \( x \) is the observed noisy image, x₀ is the true image, and n is the multiplicative speckle noise.

### Statistical Properties of Speckle Noise 📊

The statistical properties of this noise depend on the number of "looks" (L) used in processing the SAR data. For Cassini SAR, this "looks" information is available as metadata for each image. The speckle follows a Gamma distribution with mean 1 and variance 1/L:

p(n) = (L^L * n^(L-1) * e^(-Ln))/Γ(L)

(see [2] for more details). This presents an interesting challenge: we only have access to noisy images without clean ground truth. However, we know:

- The precise statistical distribution of the noise (Gamma) 
- The parameters of that distribution (from the looks metadata)
- The multiplicative relationship between noise and true image 
- "Cleaner" SAR images of Earth – getting to this quality of imagery would be a big win!

This knowledge allows us to potentially train denoising models even without paired clean/noisy data by leveraging the noise characteristics to constrain the solution space (e.g. as in [3]).

## References 📚

- [1] [Insights into Titan's geology and hydrology based on enhanced image processing of Cassini RADAR data](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2013JE004584)
- [2] [Transformer-based SAR Image Despeckling](https://arxiv.org/abs/2201.09355)
- [3] [Self-supervised Speckle Reduction GAN for Synthetic Aperture Radar](https://ieeexplore.ieee.org/abstract/document/9455273)
