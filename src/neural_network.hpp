// Bildet den Orchestrator für die Layers, Activation, Softmax und Loss - Funktionen
#pragma once

NeuralNetwork net(
    {784, 128, 32, 10},
    ActivationType::ReLU,
    LossType::CrossEntropy
);
