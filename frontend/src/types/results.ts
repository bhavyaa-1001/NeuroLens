// Matches SHAP-style combined output
export type InferenceResult = {
  image: string;
  imageUrl?: string;
  loss_box: number;
  loss_dfl: number;
  loss_class: number;
  most_impactful_feature: string;
  impact_percent: number; // e.g., 67.3
};



