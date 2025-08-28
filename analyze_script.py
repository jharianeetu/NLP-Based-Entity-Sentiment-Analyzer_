import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Create OpenAI client with your API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_script(transcript):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI that extracts entities from transcripts and classifies them into the following categories: "
                "names, places, projects, things, and topics. "
                "Also provide an overall sentiment of the transcript: positive, negative, or neutral. "
                "Respond ONLY with two sections labeled 'Entities:' and 'Sentiment:' as shown below. "
                "If a category has no entities, leave it empty but include the category name."
            )
        },
        {
            "role": "user",
            "content": (
                f"Extract and classify all entities from the following transcript:\n\n{transcript}\n\n"
                "Format your response exactly as:\n\n"
                "Entities:\n"
                "Names:\n- name1\n- name2\nPlaces:\n- place1\nProjects:\n- project1\nThings:\n- thing1\nTopics:\n- topic1\n\n"
                "Sentiment:\npositive / negative / neutral"
            )
        }
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=700,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    print("Enter your transcript (press Enter twice to submit):")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    transcript = "\n".join(lines)

    result = analyze_script(transcript)

    print("\nAnalysis Result:\n")
    print(result)
