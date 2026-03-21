
import os
import sys
from openai import OpenAI

# Configuration
# GitHub Models uses an OpenAI-compatible API
GITHUB_API_KEY = os.getenv("GITHUB_TOKEN") # GitHub automatically provides a GITHUB_TOKEN
GITHUB_API_BASE = os.getenv("GITHUB_API_URL", "https://api.github.com/copilot_internal/v2/openai/") # Default for GitHub Models

INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "azalyst_output.log"
OUTPUT_FILE = "deepseek_recommendations.md"

def analyze_with_deepseek(content):
    client = OpenAI(
        api_key=GITHUB_API_KEY,
        base_url=GITHUB_API_BASE,
    )

    try:
        completion = client.chat.completions.create(
            model="deepseek-v3", # Or deepseek-r1 if you prefer
            messages=[
                {"role": "system", "content": "You are an expert developer. Analyze the following output from the Azalyst-ETF-Intelligence GitHub Action. Provide concise recommendations for improvements, potential issues, or next steps based on the log content. Focus on actionable advice related to the ETF intelligence system's operation, data fetching, classification, or reporting."},
                {"role": "user", "content": f"Here is the Azalyst-ETF-Intelligence workflow output to analyze:\n\n{content}"}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error calling GitHub Models API: {e}"

if __name__ == "__main__":
    if not GITHUB_API_KEY:
        print("Error: GITHUB_TOKEN not set. Ensure the workflow has permissions: contents: read.")
        sys.exit(1)

    try:
        with open(INPUT_FILE, "r") as f:
            output_content = f.read()
        
        print(f"Analyzing output from {INPUT_FILE} with DeepSeek via GitHub Models...")
        recommendations = analyze_with_deepseek(output_content)
        
        print("\n### DeepSeek Recommendations ###")
        print(recommendations)
        
        with open(OUTPUT_FILE, "w") as f:
            f.write("### DeepSeek Recommendations for Azalyst-ETF-Intelligence ###\n\n")
            f.write(recommendations)
        print(f"Recommendations saved to {OUTPUT_FILE}")
            
    except FileNotFoundError:
        print(f"Error: Input file {INPUT_FILE} not found. Ensure the previous step redirects its output to this file.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
