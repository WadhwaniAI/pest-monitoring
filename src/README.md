### Source Documentation

Our codebase is powered by the pytorch lightning framework which uses PyTorch underneat it. The different components are present under the folders as follows,
1. `data/`: Contains all the data-related components (DataLoaders, Dataset, Transforms)
2. `network/`: Contains the networks of different types (Object Detection, Object Detection + Classification)
3. `callbacks/`: Contains different callbacks (Saving Predictions, Metric Calculation)
4. `metrics/`: Contains metrics evaluation code for object detection.
5. `models/`: Contains different `LightningModule`'s to package the `network`, `datamodules`, `optimizers` together for the training/inference process.
6. `utils/`: Contains utility functions to support object-detection like tasks.
7. Deprecated `loss/`: Contains code for generic loss computation.
