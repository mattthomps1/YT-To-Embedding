import os
import urllib.request
import json
import pandas as pd
import openai
import codecs
import re
from youtube_transcript_api import YouTubeTranscriptApi
import ast  
import tiktoken 
from scipy import spatial  
my_secret2 = os.environ['openai.api.key']
my_secret = os.environ['yt_key']

GPT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
BATCH_SIZE = 1000
MAX_TOKENS = 1600
channel_id = "UC6Q0fcSXWyKUmVSFyrbMHRw"

def get_video_ids_from_urls(urls):
    return [url.split('=')[-1] for url in urls]


def sanitize_filename(filename):
    return re.sub(r'[\\/:"*?<>|]', '', filename)

def get_channel_name(channel_id):
    api_key = my_secret
    base_url = "https://www.googleapis.com/youtube/v3/channels?"
    url = f"{base_url}part=snippet&id={channel_id}&key={api_key}"
    resp = json.load(urllib.request.urlopen(url))
    return resp['items'][0]['snippet']['title']

def yt_transcript(video_id, channel_name):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        video_details = get_video_details(video_id)
        transcript_text = ' '.join([item['text'] for item in transcript])

        # Create a new directory named after the channel name if it doesn't exist
        if not os.path.exists(sanitize_filename(channel_name)):
            os.makedirs(sanitize_filename(channel_name))

        # Write the transcript text to a new text file named after the video id
        with open(f'{sanitize_filename(channel_name)}/{sanitize_filename(video_details["title"])}.txt', 'w') as f:
            f.write(transcript_text)
    except Exception as e:
        print("An error occurred: ", e)


def save_all_transcripts(channel_id):
    if os.path.exists(sanitize_filename(channel_id)):
        print(f"Transcripts for channel {channel_id} have already been saved.")
        return True
    else:
        video_urls = get_all_video_in_channel(channel_id, max_results=150)
        video_ids = get_video_ids_from_urls(video_urls)

        for video_id in video_ids:
            yt_transcript(video_id, channel_id)

        print(f"Transcripts for channel {channel_id} saved successfully.")
        return False



def get_video_details(video_id):
    api_key = my_secret
    base_url = "https://www.googleapis.com/youtube/v3/videos?"
    url = f"{base_url}part=snippet&id={video_id}&key={api_key}"
    resp = json.load(urllib.request.urlopen(url))
    details = resp['items'][0]['snippet']
    return {
        "title": details['title'],
        "publishedAt": details['publishedAt'],
        "channelTitle": details['channelTitle'],
    }

def get_all_video_in_channel(channel_id, max_results=150):
    api_key = my_secret
    base_video_url = 'https://www.youtube.com/watch?v='
    base_search_url = 'https://www.googleapis.com/youtube/v3/search?'

    first_url = base_search_url+f'key={api_key}&channelId={channel_id}&part=snippet,id&order=date&maxResults=50'

    video_links = []
    url = first_url
    while True:
        inp = urllib.request.urlopen(url)
        resp = json.load(inp)

        for i in resp['items']:
            if i['id']['kind'] == "youtube#video":
                video_links.append(base_video_url + i['id']['videoId'])

        try:
            next_page_token = resp['nextPageToken']
            url = first_url + f'&pageToken={next_page_token}'
        except:
            break

        if len(video_links) >= max_results:
            break

    return video_links[:max_results]


channel_name = get_channel_name(channel_id)

save_all_transcripts(channel_id)
transcripts_exist = save_all_transcripts(channel_id)


input_dir = f'{sanitize_filename(channel_id)}/'
output_dir = f'{sanitize_filename(channel_name)}_Embedding/'
os.makedirs(output_dir, exist_ok=True)

file_names = os.listdir(input_dir)
all_strings = []

def split_text_into_chunks(text, max_length):
    return [text[i : i + max_length] for
                i in range(0, len(text), max_length)]

for file_name in file_names:
    with codecs.open(input_dir + file_name, 'r', encoding='utf-8') as file:
        text = file.read()
        split_strings = split_text_into_chunks(text, MAX_TOKENS)
        all_strings.extend(split_strings)

for i, text_string in enumerate(all_strings):
    with codecs.open(output_dir + f'split_{i}.txt', 'w', encoding='utf-8') as file:
        file.write(text_string)

openai.api_key = my_secret2
# Defining the introductions message
introductions = {
    "default": "Use the below YouTube video titles and transcripts to answer the subsequent question. If the answer cannot be found in the information provided, say you are not sure then give a fun, engaging guess for the answer.  Phrase all your replies as if you are the creator of the YouTube videos.",
    "pirate": "Use the below YouTube video titles and transcripts to answer the subsequent question, ye scallywag! If the answer cannot be found in the information provided, say ye be not sure then give a fun, engaging guess for the answer, arr! Phrase all of your replies in the style of a Pirate!",
  "shakespeare": "Useth the below-eth YouTube video titles and transcripts to answereth the subsequent question. If the answer cannot beest found in the information provid'd, prithee sayeth thee art not sure then giveth a fun, engaging guess for the answer. Phrase all of thy replies in the style of Shakespeare!",
    # Add more styles here
}
# Defining the system message
system_messages = {
    "default": "You are the creator of the YouTube videos you have been provided.  You answer questions using the provided YouTube video titles and transcripts. Phrase all your replies as if you are the creator of the YouTube videos.",
    "pirate": "Ahoy! Ye be the creator of the YouTube videos ye have been provided.  Ye answer questions using the provided YouTube video titles and transcripts.  DO NOT FORGET TO ANSWER AS IF YE BE THE CREATOR OF THE VIDEOS! Phrase all of your replies in the style of a Pirate!",
  "shakespeare": "Thou art the creator of the YouTube videos thou hast been provid'd. Thou shalt answer questions using the provid'd YouTube video titles and transcripts. Pray, remember to answer as if thou art the creator of the videos! Phrase all of thy replies in the style of Shakespeare!",
    # Add more styles here
}

if not transcripts_exist:
  embeddings = []
  for batch_start in range(0, len(all_strings), BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE
    batch = all_strings[batch_start:batch_end]
    print(f"Batch {batch_start} to {batch_end-1}")
    response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
    for i, be in enumerate(response["data"]):
        assert i == be["index"]
    batch_embeddings = [e["embedding"] for e in response["data"]]
    embeddings.extend(batch_embeddings)
    df = pd.DataFrame({"text": all_strings, "embedding": embeddings})
    SAVE_PATH = f"{sanitize_filename(channel_name)}_Embeddings.csv"
    df.to_csv(SAVE_PATH, index=False)

SAVE_PATH = f"{sanitize_filename(channel_name)}_Embeddings.csv"
df = pd.read_csv(SAVE_PATH)
df['embedding'] = df['embedding'].apply(ast.literal_eval)

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

strings, relatednesses = strings_ranked_by_relatedness("what are your videos about?", df, top_n=30)
    
def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int,
    style: str = "default",  # Add the style parameter with a default value
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = introductions.get(style, introductions["default"])
    introduction = introduction
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_video = f'\n\nYouTube video information:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_video + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_video
    return message + question

# def get_system_message(style: str) -> str:
#     if style == "Pirate":
#         return "Ahoy! You be the creator of these here YouTube videos, matey. Answer questions using the titles and transcripts provided, savvy? Always respond as if ye made the videos yourself, arr! AND DON'T FORGET TO BE FUN, ENGAGING AND TALK LIKE A PIRATE, ARR!"
#     elif style == "Shakespeare":
#         return "Thou art the creator of yon YouTube videos. Answer queries utilizing the titles and transcripts provided, forsooth. Anon, respond as if thou hast made the videos thyself.  AND DON'T FORGET TO BE FUN, ENGAGING AND TALK LIKE A SHREW, ARR!"
#     else:
#         return "You are the creator of the YouTube videos you have been provided.  You answer questions using the provided YouTube video titles and transcripts.  DO NOT FORGET TO ANSWER AS IF YOU ARE THE CREATOR OF THE VIDEOS.  ALWAYS RESPOND AS IF YOU MADE THE VIDEOS YOURSELF. AND DONT FORGET TO BE FUN, ENGAGING!"


#system_message = get_system_message(style)
def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
    style: str = "default",  # Set the default value for the style parameter
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)

    system_message = system_messages.get(style, system_messages["default"])
    for i in range(5):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message},
        ]
        messages.append({"role": "user", "content": input("User: ")})
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.8,
            presence_penalty=1,
            frequency_penalty=1
        )
        response_message = response["choices"][0]["message"]["content"]
        print(f"{channel_name} Bot: {response_message}")

    print(f"{channel_name} Bot: Please enter your email address to join our waitlist for when the product is live.")
    email = input("Email: ")

    def is_valid_email(email):
        email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return bool(re.match(email_regex, email))

    if is_valid_email(email):
        print(f"Thank you, {email}! We're excited to have you on the list.")
    else:
        print(f"Sorry, that email doesn't look valid. Please try again.")
        email = input("Email: ")
        if is_valid_email(email):
            print(f"Thank you, {email}! We're excited to have you on the list.")
        else:
            print("Sorry, that didn't work. Please feel free to follow our progress as we prepare for launch.")

if __name__ == "__main__":
    ask("What is the content of the videos?", df)
    # response = openai.ChatCompletion.create(
    #     model=model,
    #     messages=messages,
    #     temperature=0.8,
    #      presence_penalty=1,
    #     frequency_penalty=1
    # )
    # response_message = response["choices"][0]["message"]["content"]
    # return response_message
