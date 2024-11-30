from flask import Flask, request, jsonify
import joblib
import pandas as pd

# 创建 Flask 应用
app = Flask(__name__)

# 加载模型
model_filename = "random_forest_model.pkl"
model = joblib.load(model_filename)
print(f"Model loaded from {model_filename}")

# 定义编码映射
gender_encoding = {"Female": 0, "Male": 1}
campaign_channel_encoding = {"Email": 0, "PPC": 1, "Referral": 2, "SEO": 3, "Social Media": 4}
campaign_type_encoding = {"Awareness": 0, "Consideration": 1, "Conversion": 2, "Retention": 3}

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

        # # 应用编码
        # try:
        #     input_data['Gender'] = input_data['Gender'].map(gender_encoding)
        #     input_data['CampaignChannel'] = input_data['CampaignChannel'].map(campaign_channel_encoding)
        #     input_data['CampaignType'] = input_data['CampaignType'].map(campaign_type_encoding)
        # except KeyError as e:
        #     return jsonify({'error': f'Invalid value in input data: {e}'}), 400
        
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
        probability = model.predict_proba(input_data)[:, 1]
        votes = [tree.predict(input_data)[0] for tree in model.estimators_]
        confidence = votes.count(prediction[0]) / len(votes)

        # 返回结果
        response = {
            'prediction': int(prediction[0]),
            # 'probability of it being 1': float(probability[0]),
            'confidence': float(confidence)
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 启动 Flask 应用
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
