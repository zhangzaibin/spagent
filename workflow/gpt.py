from openai import OpenAI
client = OpenAI(api_key="sk-STKyxNjkvs5qdc2Yg0FepKdQ4ZcvoeJTxlOzzzfXLYASVV2P", base_url="http://35.220.164.252:3888/v1/")


# 测试可以用大模型，但是批量跑实验最好提前算好token。
response = client.responses.create(
    model="gpt-4o-mini",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)