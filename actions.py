from rasa_sdk import Action
from rasa_sdk.events import SlotSet
import sqlite3

class ActionFaqAnswer(Action):
    def name(self):
        return "action_faq_answer"

    def run(self, dispatcher, tracker, domain):
        # Get the user's question from the tracker
        question = tracker.latest_message.get('text', '').strip().lower()

        # Ensure the question is not empty
        if not question:
            dispatcher.utter_message(text="Can you please clarify your question?")
            return []

        # Connect to the SQLite database
        conn = None
        try:
            conn = sqlite3.connect("faq.db")
            cursor = conn.cursor()

            # Query the database to find a matching answer
            cursor.execute("SELECT answer FROM faqs WHERE LOWER(question) LIKE ?", (f"%{question}%",))
            result = cursor.fetchone()

            if result:
                dispatcher.utter_message(text=result[0])  # Return the found answer
            else:
                dispatcher.utter_message(text="Sorry, I couldn't find an answer to that question.")  # No match found
        
        except sqlite3.Error as e:
            # Handle SQLite database errors
            dispatcher.utter_message(text="An error occurred while searching for the answer. Please try again later.")
            print(f"Database error: {e}")  # Logs the error for debugging purposes

        finally:
            if conn:
                conn.close()  # Ensure the database connection is always closed, even in case of errors

        return []
