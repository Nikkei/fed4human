AZURE_OPEN_AI_MODEL_ID2PRICING = {
    # cf. https://azure.microsoft.com/en-us/pricing/details/azure-openai/
    "gpt-5.4": {
        "price_per_input_token": 2.5 / 1_000_000,
        "price_per_output_token": 15 / 1_000_000,
    },
    "gpt-5.2": {
        "price_per_input_token": 1.75 / 1_000_000,
        "price_per_output_token": 14 / 1_000_000,
    },
    "gpt-5.1": {
        "price_per_input_token": 1.25 / 1_000_000,
        "price_per_output_token": 10 / 1_000_000,
    },
    # GPT-4o-2024-0513 Regional
    "gpt-4o": {
        "price_per_input_token": 5 / 1_000_000,
        "price_per_output_token": 15 / 1_000_000,
    },
}
