from flask import Flask, request, jsonify
import joblib
import pandas as pd

# 创建 Flask 应用
app = Flask(__name__)

# 加载模型
model_filename = "random_forest_model.pkl"
model = joblib.load(model_filename)
print(f"Model loaded from {model_filename}")

# 定义预测接口
@app.route('/predict', methods=['POST'])
def predict():
    # 获取 JSON 数据
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    try:

        # 将输入数据转换为 DataFrame
        input_data = pd.DataFrame([data])
        
        # 检查是否包含所有必要的特征
        required_features = model.feature_names_in_
        print(f"Required features: {required_features}")
        missing_features = [feature for feature in required_features if feature not in input_data.columns]
        if missing_features:
            return jsonify({'error': f'Missing features: {missing_features}'}), 400

        # 确保输入数据顺序与模型训练时一致
        input_data = input_data[required_features]

        # 模型预测
        prediction = model.predict(input_data)
        votes = [tree.predict(input_data)[0] for tree in model.estimators_]
        confidence = votes.count(prediction[0]) / len(votes)

        # 返回结果
        response = {
            'prediction': int(prediction[0]),
            'confidence': float(confidence)
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 启动 Flask 应用
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
