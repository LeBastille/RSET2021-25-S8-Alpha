from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

class MultiMoora:
    def __init__(self, products, criteria, weights, beneficial):
        self.products = products
        self.criteria = criteria
        self.weights = np.array(weights)
        self.beneficial = np.array(beneficial, dtype=bool)
        self.matrix = self.convert_to_numeric()

    def convert_to_numeric(self):
        categorical_mapping = {
            "Snapdragon 8 Gen 2": 9,
            "Snapdragon 8 Gen 1": 8,
            "MediaTek Dimensity 9200": 7,
            "OLED": 9,
            "AMOLED": 8,
            "LCD": 5,
        }
        matrix = []
        for specs in self.products.values():
            row = []
            for value in specs.values():
                if isinstance(value, str) and value in categorical_mapping:
                    row.append(categorical_mapping[value])
                elif isinstance(value, (int, float)):
                    row.append(value)
                else:
                    raise ValueError(f"Invalid data type for value: {value}")
            matrix.append(row)
        return np.array(matrix, dtype=float)

    def normalize_matrix(self):
        return self.matrix / np.sqrt((self.matrix ** 2).sum(axis=0))

    def apply_moora(self):
        norm_matrix = self.normalize_matrix()
        weighted_matrix = norm_matrix * self.weights
        beneficial_scores = (weighted_matrix * self.beneficial).sum(axis=1)
        non_beneficial_scores = (weighted_matrix * ~self.beneficial).sum(axis=1)
        moora_scores = beneficial_scores - non_beneficial_scores
        rankings = np.argsort(-moora_scores)
        return rankings, moora_scores

    def compare_products(self):
        rankings, scores = self.apply_moora()
        result = "\nProduct Comparison using Multi-MOORA:\n"
        for rank, index in enumerate(rankings):
            product_name = list(self.products.keys())[index]
            result += f"{rank + 1}. {product_name} - Score: {scores[index]:.3f}\n"

        best_product = list(self.products.keys())[rankings[0]]
        second_best_product = list(self.products.keys())[rankings[1]]
        best_specs = self.products[best_product]
        second_best_specs = self.products[second_best_product]

        advantages = []
        for criterion, best_value, second_value in zip(self.criteria, best_specs.values(), second_best_specs.values()):
            if isinstance(best_value, (int, float)) and best_value > second_value:
                advantages.append(f"{criterion} ({best_value} vs {second_value})")

        if advantages:
            result += f"\n{best_product} stands out with better: " + ", ".join(advantages) + ".\n"
        
        return result

products = {
    "Phone A": {"Processor": "Snapdragon 8 Gen 2", "Battery Life": 5000, "Screen": "AMOLED", "Camera": 108},
    "Phone B": {"Processor": "Snapdragon 8 Gen 1", "Battery Life": 5000, "Screen": "OLED", "Camera": 50}
}

criteria = ["Processor", "Battery Life", "Screen", "Camera"]
weights = [0.4, 0.3, 0.2, 0.1]
beneficial = [True, True, True, True]

moora = MultiMoora(products, criteria, weights, beneficial)

@app.route('/compare', methods=['POST'])
def compare():
    data = request.json
    if data['product1'] in products and data['product2'] in products:
        result = moora.compare_products()
        return jsonify({"result": result})
    else:
        return jsonify({"error": "Invalid product selection"}), 400

if __name__ == '__main__':
    app.run(debug=True)
