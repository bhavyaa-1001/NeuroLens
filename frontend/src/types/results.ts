export type InferenceResult = {
  imageName: string;
  imageUrl?: string;
  predictedLoss: number;
  category: 1 | 2 | 3 | 4;
  parametricDescription: string;
};


