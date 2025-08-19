# ARTIST (Modified Version for Thesis)

This folder contains a **modified copy** of the [ARTIST](https://github.com/ARTIST-Association/ARTIST)  
(**AI-enhanced differentiable Ray Tracer for Irradiation Prediction in Solar Tower Digital Twins**).

---

## Origin
- **Project:** ARTIST — AI-enhanced differentiable Ray Tracer  
- **Authors:** ARTIST Association (see original repository for contributors)  
- **License:** MIT (see original LICENSE file)  
- **Documentation:** [https://artist.readthedocs.io](https://artist.readthedocs.io)

---

## Purpose in This Repository
This modified version is an **integral part of the training loop**.  
Although newer versions of ARTIST are available open source, the pipeline is coupled to this specific release with minor changes, so it is included here for **reproducibility**.

---

## Modifications
I applied a few small changes to integrate ARTIST with my pipeline. In particular:
- Adjusted function calls and outputs in the raytracing loop to match my training setup.  
- Modified I/O to exchange data more easily with the neural network components.  
- Minor clean-ups (paths, logging).  

---

## Usage in the Thesis Pipeline
- The `src/pipeline/` code imports and uses this version directly.  
- Replacing it with a newer upstream release will not work without adapting the loop.

---

## Acknowledgment
Full credit goes to the original ARTIST developers and contributors.  
This repo only redistributes a lightly adapted copy to ensure reproducibility of my thesis results.
