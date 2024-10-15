#include <iostream>
#include <iomanip>  // For setting decimal precision
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"

// This function will act as a signal wrapper for your features array
static int get_signal_data(size_t offset, size_t length, float *out_ptr, float *feature) {
    memcpy(out_ptr, feature + offset, length * sizeof(float));
    return 0;
}

int main() {
    // Array of features (input data between 0 and 4000)
    float features[] = {500, 1500, 2500, 3500, 4000};

    // Set the precision to 6 decimal places
    std::cout << std::fixed << std::setprecision(6);

    // Iterate over each feature in the array and classify individually
    for (int i = 0; i < sizeof(features) / sizeof(features[0]); i++) {
        // Prepare input for each classification (1D input expected)
        float input_feature[1] = {features[i]};
        
        // Wrap the feature in a signal structure
        ei::signal_t signal;
        signal.total_length = 1;  // Single feature input
        signal.get_data = [&](size_t offset, size_t length, float *out_ptr) {
            return get_signal_data(offset, length, out_ptr, input_feature);
        };

        // Result object to store classification output
        ei_impulse_result_t result = { 0 };

        // Run the classifier for the current feature
        EI_IMPULSE_ERROR ei_status = run_classifier(&signal, &result, false);

        if (ei_status == EI_IMPULSE_OK) {
            // Print the feature value being classified
            std::cout << "Feature: " << features[i] << std::endl;
            std::cout << "Classification results:" << std::endl;

            // Print the classification results with readable precision
            for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
                std::cout << " - " << result.classification[ix].label << ": " 
                          << result.classification[ix].value << std::endl;
            }

            std::cout << std::endl;  // Add spacing between results
        } else {
            // Handle classification error
            std::cerr << "Error running classifier: " << ei_status << std::endl;
        }
    }

    return 0;
}
