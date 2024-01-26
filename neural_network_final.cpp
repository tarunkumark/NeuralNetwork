#include <iostream>
#include <math.h>
#include <vector>
#include <cmath>
#include <random>

using namespace std;

double random_num(){
    double mean = 0.0;
    double std_dev = 1.0;

    // Seed the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Create a normal distribution with the given mean and standard deviation
    std::normal_distribution<double> normal_dist(mean, std_dev);

    // Generate a random number from the normal distribution
    double random_number = normal_dist(gen);
    return random_number;
}


double sigmoid(double x){
    return 1/(1+exp(-x));
}

double deriv_sigmoid(double x){
    double fx = sigmoid(x);
    return fx*(1-fx);
}

std::vector<double> square_array(std::vector<double> v1, std::vector<double> v2){
    std::vector<double> sv;
    for(int i = 0 ; i < v1.size(); i++){
        sv.push_back(pow(v1[i] - v2[i],2));
    }
    return sv;
}

double mean(std::vector<double> v){
    double mean = 0;
    for(int i = 0; i < v.size(); i++){
        mean+=v[i];
    }
    return mean/v.size();
}

double mean_squared_error(std::vector<double> v1,std::vector<double> v2){
    return mean(square_array(v1,v2));
}

double mse_loss(std::vector<double>& y_true, std::vector<double>& y_pred) {
        double sum = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            sum += pow(y_true[i] - y_pred[i], 2);
        }
        return sum / y_true.size();
    }

double dot_product(std::vector<double> v1, std::vector<double> v2){
    double sum = 0;
    for(int i = 0; i < v1.size(); i++){
        sum+=v1[i]*v2[i];
    }
    return sum;
}

class Neuron{
    std::vector<double> weights;
    double bias;
    public:
        Neuron(std::vector<double>w, double b){
            bias = b;
            weights = w;
        }
        double feed_forward(std::vector<double>inputs){
            return sigmoid(dot_product(inputs,weights)+bias);
        }
};

class NeuralNetwork{
    
    double w1;
    double w2;
    double w3;
    double w4;
    double w5;
    double w6;

    double b1;
    double b2;
    double b3;
    public:
        NeuralNetwork() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<double> dist(0.0, 1.0);
             w1 = dist(gen);
             w2 = dist(gen);
             w3 = dist(gen);
             w4 = dist(gen);
             w5 = dist(gen);
             w6 = dist(gen);

             b1 = dist(gen);
             b2 = dist(gen);
             b3 = dist(gen);
        }
        double feed_forward(std::vector<double> x){
            double h1 = sigmoid(w1 * x[0] + w2 * x[1] + b1);
            double h2 = sigmoid(w3 * x[0] + w4 * x[1] + b2);
            double o1 = sigmoid(w5 * h1 + w6 * h2 + b3);
            return o1;
        }
        
        void train(std::vector<std::vector<double>> data,std::vector<double> all_y_values){
            double learn_rate = 0.1;
            long int epochs = 1000;
            for(long int i = 0; i < epochs;i++){
                for(int j = 0; j < all_y_values.size(); j++){
                    double sum_h1 = w1 * data[j][0] + w2 * data[j][1] + b1;
                    double h1 = sigmoid(sum_h1);

                    double sum_h2 = w3 * data[j][0] + w4 * data[j][1] + b2;
                    double h2 = sigmoid(sum_h2);

                    double sum_o1 = w5 * h1 + w6 * h2 + b3;
                    double o1 = sigmoid(sum_o1); 
                    double y_pred = o1;

                    double d_L_d_ypred = -2 * (all_y_values[j] - y_pred);

                    double d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1);
                    double d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1);
                    double d_ypred_d_b3 = deriv_sigmoid(sum_o1);

                    double d_ypred_d_h1 = w5 * deriv_sigmoid(sum_o1);
                    double d_ypred_d_h2 = w6 * deriv_sigmoid(sum_o1);

                    double d_h1_d_w1 = data[j][0] * deriv_sigmoid(sum_h1);
                    double d_h1_d_w2 = data[j][1] * deriv_sigmoid(sum_h1);
                    double d_h1_d_b1 = deriv_sigmoid(sum_h1);

                    double d_h2_d_w3 = data[j][0] * deriv_sigmoid(sum_h2);
                    double d_h2_d_w4 = data[j][1] * deriv_sigmoid(sum_h2);
                    double d_h2_d_b2 = deriv_sigmoid(sum_h2);

                    w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1;
                    w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2;
                    b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1;

                    w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3;
                    w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4;
                    b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2;

                    w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5;
                    w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6;
                    b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3;
                }
                if(i % 10 == 0){
                        std::vector<double> y_preds;
                        for(std::vector<double> datum: data){
                            y_preds.push_back(feed_forward(datum));
                        }
                        double loss = mse_loss(all_y_values, y_preds);
                        cout << "Epoch " << i << " loss: " << loss << endl;

                    }
            }
        }
        
};

int main() {
    NeuralNetwork network = NeuralNetwork();
    std::vector<std::vector<double>> data = {{-2,-1},{25,6},{17,4},{-15,-6}};
    std::vector<double> inputs = {1,0,0,1};
    network.train(data,inputs);
    std::vector<double> emily = {-7,-3};
    std::vector<double> frank = {20,2};
    cout << "Emily:" << network.feed_forward(emily) << endl;
    cout << "Frank:" << network.feed_forward(frank)<< endl;
    return 0;
}

