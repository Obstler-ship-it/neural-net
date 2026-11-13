//Implementierung für Ada Gradientenabstieg
#pragma once

#include <algorithm>
#include <cmath>
#include <exception>
#include <vector>
#include <stdexcept>
#include "../matrix/matrix.hpp"

class Adam{
    public:

    struct Config {
        float learning_rate = 0.01f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
    };

    explicit Adam() : Adam(Config{}) {} //Default Konstruktor mit den Werten oben im struct

    explicit Adam(const Config& config) //Konstruktor mit setter für Config
        : config(config) {
            if (!(config.learning_rate > 0))
                throw std::invalid_argument("Adam: learning_rate must be > 0");

            if (!(config.beta1 >= 0 && config.beta1 < 1))
                throw std::invalid_argument("Adam: beta1 must be in [0, 1)");

            if (!(config.beta2 >= 0 && config.beta2 < 1))
                throw std::invalid_argument("Adam: beta2 must be in [0, 1)");
        }

    void step(Matrix<Layout::ColumnMajor>& dW, Matrix<Layout::ColumnMajor>& db){
        if (t==0){

            //Initialisierung s_t
            st_dW = dW.elemntweisemult(dW) * (1 - config.beta2);
            st_db = db.elemntweisemult(db) * (1 - config.beta2);
                
            //Initialisierung m_t
            mt_dW = dW * (1 - config.beta1);
            mt_db = db * (1 - config.beta1);
                
            dW * (-config.learning_rate/(1 - config.beta1));
            db * (-config.learning_rate/(1 - config.beta1));
        } else {

            (st_dW * config.beta2) += (dW.elemntweisemult(dW) * (1 - config.beta2));
            (st_db * config.beta2) += (db.elemntweisemult(db) * (1 - config.beta2));

            (mt_dW * config.beta1) += (dW * (1 - config.beta1));
            (mt_db * config.beta1) += (db * (1 - config.beta1));

            dW = -config.learning_rate * ((mt_dW * (1.0f/(1-std::pow(config.beta1, t)))) / (st_dW * (1.0f/(1-std::pow(config.beta2, t)))).sqrt())
            db = -config.learning_rate * ((mt_db * (1.0f/(1-std::pow(config.beta1, t)))) / (st_db * (1.0f/(1-std::pow(config.beta2, t)))).sqrt())
        }
    }

    private:
         const Config& config;
         Matrix<Layout::ColumnMajor> mt_dW {0,0 ,Initialization::Uninitialized};
         Matrix<Layout::ColumnMajor> mt_db {0,0 ,Initialization::Uninitialized};

         Matrix<Layout::ColumnMajor> st_dW {0,0 ,Initialization::Uninitialized};
         Matrix<Layout::ColumnMajor> st_db {0,0 ,Initialization::Uninitialized};

         size_t t = 0; //Ist nur ein size_t aber eigentlich benötigt nicht jedes Adam Objekt ein eigenes

}
