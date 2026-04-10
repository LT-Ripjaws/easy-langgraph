import streamlit as st
from backend import bot
from langchain_core.messages import HumanMessage

seed = '1'
CONFIG = {'configurable': {'thread_id': seed}}

# st.session_state -> dict ->
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = [] # adding a key with a new list as value in the session dict

# loading convo history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input =  st.chat_input('Type here..')

if user_input:
    st.session_state['message_history'].append({'role':'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

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