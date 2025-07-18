# tracET: A Software for Tracing Low-Level Structures in Cryo-Electron Tomography

## Installation

* Clone the repository: [https://github.com/PelayoAlvarezBrecht/tracer/](https://github.com/PelayoAlvarezBrecht/tracer/)
* Once inside, make sure all the packages listed in [requirements.txt](https://github.com/PelayoAlvarezBrecht/tracer/tree/pypi/requirements.txt) are installed by running:

  ```bash
  pip install -r requirements.txt
  ```
* Finally, in a terminal, run the command:

  ```bash
  python3 tracET/setup.py install
  ```

## Scripts

There are six different scripts used to apply different parts of the process:

---

### Saliency Map

* **Script description:**

  * The name of the script is `get_saliency.py`.
  * From a tomogram with a binary segmentation, it calculates the saliency map, a distance transformation of the input smoothed with a Gaussian filter.
  * This step is also included in the `apply_nonmaxsup.py` script.

* **Parameters:**

  * The parameter `in_tomo`, called with `-i` or `--itomo`, specifies the name of the input file, a binary map tomogram in MRC or NRRD format.
  * The parameter `smooth_deviation`, called with `-s` or `--sdesv`, sets the deviation for the Gaussian filter. It should be approximately 1/3 of the element radius.

* **Outputs:**

  * A tomogram with the saliency map, in the same format as the input and with the suffix `_saliency`.

---

### Non-Maximum Suppression

* **Script description:**

  * The name of the script is `apply_nonmaxsup.py`.
  * From a segmentation or saliency map, it detects the most central voxels of the elements and constructs an equi-spatial point cloud of these elements.

* **Parameters:**

  * `in_tomo` (`-i`, `--itomo`): Input file name, a scalar or binary map tomogram in MRC or NRRD format.
  * `smooth_deviation` (`-s`, `--sdesv`): Deviation for the Gaussian filter (\~1/3 of the element radius).
  * `skel_mode` (`-m`, `--mode`): Structural mode for computing the skeleton: `s` for surfaces, `l` for lines, and `b` for blobs.
  * `binary_input` (`-b`, `--ibin`): Set to `0` for scalar map, `1` for binary map. If `1`, a distance transformation saliency map is calculated.
  * `filter` (`-f`, `--filt`): Filter for the suppression mask. Optional. If not given, only negative values are eliminated.
  * `downsample` (`-d`, `--downs`): If provided, applies downsampling with the given radius.

* **Outputs:**

  * A tomogram containing only the saliency map maxima, in the same format as the input, with the suffix `_supred`.

---

### Spatial Embedded Graph

* **Script description:**

  * The name of the script is `trace_graph.py`. Run it with:

    ```bash
    trace_graph - options
    ```
  * From a point cloud of filaments, it traces a spatially embedded graph, calculates the different connected components and sub-branches, and models each branch as a curve to measure various properties.

* **Parameters:**

  * `input` (`-i`, `--itomo`): The tomogram containing the point cloud from filament segmentation (output of the previous script), in MRC or NRRD format.
  * `radius` (`-r`, `--rad`): Radius for connecting points in the graph.
  * `subsampling` (`-s`, `--subsam`): Radius used for point subsampling. If not given, no subsampling is applied.

* **Outputs:**

  * A **VTP file** with graph component, branch, and geometric data: same name as input + `_skel_graph.vtp`.
  * A **CSV file** with the same information: same name as input + `_skel_graph.csv`.

---

### Blobs Clustering

* **Script description:**

  * The name of the script is `Get_cluster.py`. Run it with:

    ```bash
    get_cluster - options
    ```
  * From a point cloud tomogram of blobs, it clusters the points using MeanShift or Affinity Propagation and localizes the centroids.

* **Parameters:**

  * `input` (`-i`, `--itomo`): The tomogram with the point cloud from filament segmentation, in MRC or NRRD format.
  * `mode` (`-m`, `--mode`): Clustering algorithm to use:

    * `Affinity` uses Affinity Propagation (only recommended for small tomograms).
    * `MeanShift` uses the Mean Shift algorithm (recommended for all types of tomograms). Requires two additional parameters:

      * `blob_diameter` (`-b`, `--blob_d`): Diameter of the blobs to detect.
      * `n_jobs` (`-n`, `--n_jobs`): Number of parallel jobs for the algorithm.

* **Outputs:**

  * A **VTP file** with ribosome points labeled by their cluster: same name + `mode_labeled.vtp`.
  * An **MRC file** with the same labels: same name + `mode_labeled.mrc`.
  * A **TXT file** (convertible to IMOD *.mod* format) with centroid information for each cluster.

---

### Membrane Classification

* **Script description:**

  * The name of the script is `membrane_poly.py`. Run it with:

    ```bash
    membrane_poly - options
    ```
  * From a point cloud of membranes, it clusters points into different membranes.

* **Parameters:**

  * `in_tomo` (`-i`, `--itomo`): The tomogram with the membrane segmentation point cloud, in MRC or NRRD format.
  * `distance_clustering` (`-d`, `--dist`): Distance threshold for points to be part of the same cluster.
  * `min_samples` (`-s`, `--samp`): Minimum samples required to form a cluster. Optional. Defaults to 2 if not given.

* **Outputs:**

  * A **VTP file** with membrane points labeled by cluster (membrane): same name + `.vtp`.

---

### DICE Metric

* **Script description:**

  * The name of the script is `seg_skel_dice.py`. Run it with:

    ```bash
    seg_skel_dice - options
    ```
  * From two binary segmentations, it calculates TS, TP, and DICE metrics, and outputs the skeletons of both inputs.

* **Parameters:**

  * `in_tomo` (`-i`, `--itomo`): Input binary segmentation tomogram in MRC or NRRD format.
  * `gt_tomo` (`-g`, `--igt`): Ground truth binary tomogram in MRC or NRRD format.
  * `skel_mode` (`-m`, `--mode`): Structural mode: `s` for surfaces, `l` for lines, `b` for blobs.
  * `dilation` (`-d`, `--dil`): Number of iterations for pre-dilation. Optional. If not given, no dilation is applied.
  * `ifilter` (`-f`, `--ifilt`): Threshold for input mask filtering in non-maximum suppression. Optional. Default is 0.065; decrease if too strong.
  * `gtfilter` (`-F`, `--tfilt`): Threshold for ground truth mask filtering. Optional. Default is 0.065; decrease if too strong.

* **Outputs:**

  * **TS metric** value.
  * **TP metric** value.
  * **DICE metric** value.
  * (Optional) Skeleton of the input tomogram (`-o`, `--otomo`).
  * (Optional) Skeleton of the ground truth tomogram (`-t`, `--ogt`).

## Tutorials
In this section we are going to explain how to generate a result from a example data, that we save in the following ![directory](https://zenodo.org/records/13921453).

### Surface example: Membranes
* In the subdirectory `Membrane` we look at the file `tomo_001_mem_croped.mrc`. In paraview it looks like:

![Membrane segmentation](images_tutorial/tutorial_full_membrane.gif)

* For create the skeleton use the command with the next options: 
```commandline
apply_nonmaxsup -i tomo_001_mem_croped.mrc -s 2 -m s -b 1 -f 0.065 -d 10
```
* the result will be:

![Membrane skeleton](images_tutorial/Tutorial_supred_membrane.gif)

* Finnally we divide in different membranes using the command with the next options:
```commandline
membrane_poly -i tomo_001_mem_croped_supred.mrc -d 10
```

* The result in that case is:

![Membrane_test](images_tutorial/Tutorial_clust_membrane.png)

* With this we can get the point cloud of the membrane with the membranes separated.

### Blob example: Ribosomes
* In the subdirectory `Membrane` we look at the file `tomo_001_mem_croped.mrc`. In paraview it looks like:

![Ribo_segmentation](images_tutorial/tutorial_full_ribo.gif)

* For create the skeleton we use the command with the next options:
```commandline
apply_nonmaxsup -i tomo_001_ribo_croped.mrc -s 2 -m b -b 1 -f 0.065 -d 10
```

* The result will be:

![Ribo_skeleton](images_tutorial/tutorial_supred_ribo.gif)

* Finally, to cluster the diferent ribosomes, you just need to use the command with the next options:
```commandline
get_cluster -i tomo_001_ribo_croped_supred_time.mrc -m MeanShift -b 20 -n 20

```

![Ribo_test](images_tutorial/tutorial_cluster_ribo.gif)

* With this we already separate the ribosomes between them.

### Filamen example: Actin
* In the subdirectory `Actin` we look at the file `tomo_001_actin_trimmed.mrc`. This we show it in IMOD because is too dense to watch it in 3D:

![Actin_segmentation](images_tutorial/tutorial_full_actin.gif)

* For create the skeleton we use the command with the next options:
```commandline
apply_nonmaxsup -i tomo_001_actin_trimmed.mrc -s 2 -m b -b l -f 0.1 -d 0
```

* The result in IMOD will be:

![Actin_skeleton](images_tutorial/tutorial_supred_actin.gif)

* Finally, to get the curves and the graph, we execute:
```commandline
trace_graph -i tomo_001_actin_trimmed_supred.mrc -r 15 -s 2 -t n
```

* The result of the graph in paraview is:

![Actin_graph](images_tutorial/tutorial_graph_actin.gif)

* And the curves are:

![Actin_curves](images_tutorial/tutorial_curves_actin.gif)

* This are the skeletons of the actin net with the first one prioritizing the connection and the second prioritizing the curvatures
