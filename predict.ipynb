{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "94222528-fc0d-4ad1-b602-b2d24b9ba8b8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Prediction: Normal_Weight\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import pickle\n",
    "import sys\n",
    "\n",
    "def load_pipeline(model_path=\"best_model.pkl\"):\n",
    "    with open(model_path, \"rb\") as f:\n",
    "        d = pickle.load(f)\n",
    "    return d\n",
    "\n",
    "def preprocess_input(input_data, pipeline):\n",
    "    # Pastikan urutan fitur sama, isi missing\n",
    "    feature_names = pipeline[\"feature_names\"]\n",
    "    numerical = pipeline[\"numerical\"]\n",
    "\n",
    "    # Jika input satu baris dict, konversi ke DataFrame\n",
    "    if isinstance(input_data, dict):\n",
    "        input_df = pd.DataFrame([input_data])\n",
    "    else:\n",
    "        input_df = input_data.copy()\n",
    "\n",
    "    # Pastikan semua kolom yang diperlukan ada\n",
    "    for col in feature_names:\n",
    "        if col not in input_df.columns:\n",
    "            input_df[col] = 0  # default 0 (untuk one-hot)\n",
    "\n",
    "    # Isi numerik yang null dengan median/mean (default: 0, boleh diganti sesuai pipeline training)\n",
    "    for col in numerical:\n",
    "        if input_df[col].isnull().any():\n",
    "            input_df[col] = input_df[col].fillna(0)\n",
    "\n",
    "    # Scaling numerik\n",
    "    scaler = pipeline[\"scaler\"]\n",
    "    input_df[numerical] = scaler.transform(input_df[numerical])\n",
    "\n",
    "    # Susun ulang kolom agar sesuai urutan saat training\n",
    "    input_df = input_df[feature_names]\n",
    "    return input_df\n",
    "\n",
    "def predict(input_data, model_path=\"best_model.pkl\"):\n",
    "    pipeline = load_pipeline(model_path)\n",
    "    input_df = preprocess_input(input_data, pipeline)\n",
    "    model = pipeline[\"model\"]\n",
    "    y_pred = model.predict(input_df)\n",
    "\n",
    "    # Decode label ke bentuk asli\n",
    "    label_encoder = pipeline[\"label_encoder\"]\n",
    "    pred_label = label_encoder.inverse_transform(y_pred)\n",
    "    return pred_label\n",
    "\n",
    "# -------- Example usage CLI ------------\n",
    "if __name__ == \"__main__\":\n",
    "    # Contoh input dict (isi sesuai fitur hasil one-hot + numerik)\n",
    "    sample = {\n",
    "        \"Age\": 22, \"Height\": 1.74, \"Weight\": 75.0, \"FCVC\": 3, \"NCP\": 3, \"CH2O\": 1, \"FAF\": 1, \"TUE\": 1,\n",
    "        # Categorical one-hot yang dipakai model\n",
    "        \"Gender_Male\": 1,\n",
    "        \"family_history_with_overweight_yes\": 1,\n",
    "        \"FAVC_yes\": 1,\n",
    "        \"SMOKE_yes\": 0,\n",
    "        \"SCC_yes\": 0,\n",
    "        \"CAEC_Often\": 1,\n",
    "        \"CAEC_Sometimes\": 0,\n",
    "        \"CAEC_Never\": 0,\n",
    "        \"CALC_Sometimes\": 1,\n",
    "        \"CALC_Never\": 0,\n",
    "        \"MTRANS_Public_Transportation\": 1,\n",
    "        \"MTRANS_Automobile\": 0,\n",
    "        \"MTRANS_Other\": 0\n",
    "    }\n",
    "    result = predict(sample, \"best_model.pkl\")\n",
    "    print(\"Prediction:\", result[0])\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
