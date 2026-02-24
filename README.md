# AI-LLM-Chatbot
This is an AI chatbot connected to a Large Language Model (LLM) with an option to read information from a PDF or to search Wikipedia. 

It is written in Python and it makes use of: 
1) Retrieval Augmented Generation (RAG) to search the PDF.
2) LangChain agents with prompt engineering to help the chatbot return the most relevant information from the PDF or information returned from Wikipedia.
3) Streamlit library to create the Web UI.

The PDF is actually a Company Brochure which tells the user information about the company name EngagePro. Eg: Company's Vision and Mission, Key Achievements and Current Focus.

If the user asked a question which is not found in the PDF, the agent will search Wikipedia website using WikipediaAPIWrapper and WikipediaQueryRun.

A screenshot of the Web UI is:

<img width="550" height="450" alt="image" src="https://github.com/user-attachments/assets/527ec48e-be8c-471d-8347-9a7b0ec1a8c4" />

If the user asked about EngagePro, it will show:

<img width="550" height="450" alt="image" src="https://github.com/user-attachments/assets/be1d291d-3387-4e46-9d7d-5f7e38a91c2d" />

If the user asked about a topic not found in the PDF, it will show:

<img width="550" height="450" alt="image" src="https://github.com/user-attachments/assets/3bddda98-7bbc-47a0-819d-4581feb18844" />

If the user asked about forbidden topics, it will show:

<img width="550" height="450" alt="image" src="https://github.com/user-attachments/assets/fb4c3929-c073-4d5e-a63d-03e85ee24860" />
