# FlyDecoder

**Author:** [kristoferUA](https://github.com/kristoferUA)

A fruit fly, a suspicious string, and a truly unreasonable number of neurons. FlyDecoder is a local Python and Pygame joke of an app: the fly inspects numeric clues, then a tiny external readout picks which decoder to try. The coffee is decorative. The connectome is real.

## What the fly does

- Tries Base64, hexadecimal, binary, and URL decoding, or decides the input is already plain text.
- Sends 24 numeric input features through the CPU FlyBrain simulator and the official MaleCNS v1.0 connectome.
- Tries another decoder if the first guess produces text that looks like it was faxed from Mars.
- Lets the external readout learn from synthetic examples and saves it locally between sessions.

The fly does not understand Base64. It sees numbers; the readout makes the actual decoder choice. The joke is biological. The simulation and connectome are real.

## Run on Windows

1. Run `install_flydecoder.bat` once. It installs Python 3.12 and the application dependencies, then downloads and verifies the three required MaleCNS files (about 1.1 GB total) and builds the local graph cache.
2. Start the app with `run.bat`. Later launches work offline.

First setup needs about 8 GB of free RAM. A whole simulated brain is not especially lightweight, even when its owner is smaller than a raisin.

## Credits and licenses

- FlyBrain simulator source: [HEREISCB/flybrain](https://github.com/HEREISCB/flybrain), © Chaitanya Benade and contributors, MIT. See [LICENSE](LICENSE).
- MaleCNS v1.0 connectome data: FlyEM at HHMI Janelia, Google Research, University of Cambridge, and MRC Laboratory of Molecular Biology; CC BY 4.0. See [MALECNS_ATTRIBUTION.md](MALECNS_ATTRIBUTION.md).

For the detailed guide in Ukrainian, see [README_UA.md](README_UA.md). No flies were asked to learn cryptography.
