import streamlit as st
from backend import bot
from langchain_core.messages import HumanMessage
import uuid

# ----------** UTILITY FUNCTIONS **------------

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['message_history'] = []
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def load_conversation(thread_id):
    state = bot.get_state(config={'configurable': {'thread_id': thread_id}}).values
    return state.get('messages', [])  # returns [] if key doesn't exist yet

# seed = '1' we have commented this out because we are now generating a unique thread id for each session using the generate_thread_id function, and storing it in the session state. This way, each user gets a unique thread id and their conversations are not mixed with others. If you want to use a fixed seed for testing purposes, you can uncomment this line and set the seed value as needed.

# ------------** SESSION SETUP **-------------

# st.session_state -> dict ->
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = [] # adding a key with a new list as value in the session dict

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id() # adding a key with a new thread id as value in the session dict

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = [] # this will be a list to store the thread ids and their corresponding message histories, so that we can display the list of conversations in the sidebar and load the selected conversation history when a thread id is clicked.

add_thread(st.session_state['thread_id']) 

# ------------** SIDEBAR UI **-------------

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(f'Chat Thread: {str(thread_id)}'):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role': role, 'content': message.content})

        st.session_state['message_history'] = temp_messages

# ------------** STREAMLIT UI **-------------

# loading conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input =  st.chat_input('Type here..')

if user_input:
    st.session_state['message_history'].append({'role':'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}} # we are passing the thread id from the session state to the bot's config, so that the bot can use it to store and retrieve the conversation history for that thread id in the memory saver checkpoint.
    
    # This is for the normal way of invoking the bot, but we will be using the stream way in the next step
    # response = bot.invoke({'messages': [HumanMessage(content=user_input)]}, config = CONFIG)
    # ai_message = response['messages'][-1].content
    # st.session_state['message_history'].append({'role':'assistant', 'content': ai_message})
    # with st.chat_message('assistant'):
    #     st.text(ai_message)

    # This is for the streaming way of invoking the bot
    with st.chat_message('assistant'):
        ai_message = st.write_stream( 
                        message_chunk.content for message_chunk, metadata in bot.stream(   # stream returns a generator that yields message chunks and metadata
                        {'messages': [HumanMessage(content=user_input)]},
                        config = CONFIG,
                        stream_mode='messages'
                    )
                ) 
    st.session_state['message_history'].append({'role':'assistant', 'content': ai_message}) # appending the final ai message to the message history after the streaming is done, so that it can be loaded in the next session