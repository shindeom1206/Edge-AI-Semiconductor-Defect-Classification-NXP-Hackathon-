 python scripts\PH3_CODES\ph3_evaluate_tflite.py
================================================================================
  HACKATHON PHASE-3 PREDICTION
  NXP Hackathon |
================================================================================
  Model       : C:\hackathon_project\PH3\export\mobilenetv2_float32.tflite
  Input shape : [  1 128 128   3]  dtype: float32
  Output shape: [ 1 11]  dtype: float32
  Image size  : (128, 128)
  Classes (11): ['BRIDGE', 'CLEAN_CRACK', 'CLEAN_LAYER', 'CLEAN_VIA', 'CMP', 'CRACK', 'LER', 'OPEN', 'OTHERS', 'PARTICLE', 'VIA']

  Found 331 images in: C:\edge-ai-defect-classification\Hackathon_phase3_prediction_dataset

  Predicting: 100%|█████████████████| 331/331 [00:05<00:00, 60.12it/s]

  Predicted : 331
  Skipped   : 0

  PREDICTION DISTRIBUTION:
  ---------------------------------------------
  BRIDGE        :   57  █████████████████████████████████████████████████████████
  CLEAN_CRACK   :   26  ██████████████████████████
  CLEAN_LAYER   :   29  █████████████████████████████
  CLEAN_VIA     :   19  ███████████████████
  CMP           :   36  ████████████████████████████████████
  CRACK         :   26  ██████████████████████████
  LER           :   30  ██████████████████████████████
  OPEN          :   32  ████████████████████████████████
  OTHERS        :   45  █████████████████████████████████████████████
  PARTICLE      :   17  █████████████████
  VIA           :   14  ██████████████

  Saved: C:\hackathon_project\PH3\results\predictions_20260226_030506.txt
  Lines: 331

  PREVIEW (first 10 lines):
  -------------------------------------------------------
  Image Name                          Predicted Class
  -------------------------------------------------------
  100.png                             CLEAN_CRACK
  101.png                             CRACK
  102.png                             BRIDGE
  103.png                             OPEN
  104.png                             OTHERS
  105.png                             PARTICLE
  106.png                             CMP
  107.png                             CLEAN_CRACK
  108.png                             BRIDGE
  109.png                             CRACK
  ... and 321 more lines

================================================================================
  DONE  --  Upload this file to the hackathon portal:
  C:\hackathon_project\PH3\results\predictions_20260226_030506.txt
================================================================================
