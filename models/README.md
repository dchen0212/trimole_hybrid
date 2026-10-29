# Model configurations and checkpoints

The [22 task configurations](../configs/tasks/) and [task index](../configs/task_index.csv) specify the fixed models, training parameters, formal seeds and prediction combinations. Use the root [training](../train.py) and [prediction](../predict.py) entry points; [input formats](../docs/USAGE.md) describe the prepared feature inputs.

- [Current final model code and configuration package](https://github.com/dchen0212/trimole_hybrid/releases/download/v2026.10.29-fixed-pipeline/Final_Model_Configuration_and_Code.zip)
- [Checkpoint manifest](../checkpoints/manifest.json)
- [Original configuration package](https://github.com/dchen0212/trimole_hybrid/releases/download/v2026.10.28/Final_Model_Configuration_and_Code.zip), retained unchanged

Checkpoint weights are not attached to these releases. The manifest describes 155 located fitted artifacts, including partial component sets. Eight tasks have no saved original estimator. Source defaults and fitted settings are distinguished in the per-task records. The package is not a complete set of fitted assets for end-to-end reproduction of all 22 reported systems.

Task-specific encoder training and the full research source remain accessible through the [paper-to-code guide](../docs/RESULT_CODE_MAP.md) and [pinned pre-cleanup snapshot](https://github.com/dchen0212/trimole_hybrid/tree/6c263364f79b8c8f21fbf482812636fb801ec233).
