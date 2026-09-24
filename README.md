Neuronales-Netz

NeuralNetwork net(
      {784, 128, 32, 10},
      ActivationType::ReLU,
      LossType::CrossEntropy
);

chmod +x start.sh
bash start.sh
