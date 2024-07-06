from chatbot_chain import get_chatbot_chain
import sys
 
print("This chatbot build using OpenAI and langchain framework")

chain = get_chatbot_chain()

def main():
    while True:
        query = input('Enter your question or  "exit" to quit: ')
 
        if query == "exit":
            print('Exiting')
            sys.exit()
 
        response = chain.invoke({"question": query})
 
        print("Answer: " + response["answer"])

if __name__=="__main__":
    main()