### remotejsondata.py

from util import p

# requests
import requests
import json

response = requests.get('https://jsonplaceholder.typicode.com/posts')
data = response.json()
p(data)

# urllib
import json
from urllib.request import urlopen

response = urlopen('https://jsonplaceholder.typicode.com/posts')
if response.getcode() == 200:
    data = json.loads(response.read().decode('utf-8'))
    for post in data:
        p(post['title'])
else:
    p('Error occurred!')

# aiohttp
import aiohttp
import asyncio
import json

async def fetch_json(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()
            return data

async def main():
    url = 'https://jsonplaceholder.typicode.com/posts'
    data = await fetch_json(url)
    p(json.dumps(data, indent=4))

asyncio.run(main())
