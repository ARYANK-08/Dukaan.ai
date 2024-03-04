from django.shortcuts import render,HttpResponse
import csv
from django.shortcuts import render

import pandas as pd
import chardet
from django.shortcuts import render

import pandas as pd
import chardet
from django.shortcuts import render
from io import StringIO  # Import StringIO

# Create your views here.
def index(request):
    return render(request, 'index.html')

def winter(request):
    return render(request, 'winter.html')

def summer(request):
    return render(request, 'summer.html')

def autumn(request):
    return render(request, 'autumn.html')

def spring(request):
    return render(request, 'spring.html')




def upload_csv(request):
    if request.method == 'POST' and request.FILES['csv_file']:
        csv_file = request.FILES['csv_file']
        
        # Check if the uploaded file is a CSV file
        if not csv_file.name.endswith('.csv'):
            return render(request, 'retailer/upload_csv.html', {'error_message': 'Please upload a CSV file.'})
        
        try:
            # Read the first few bytes of the file to detect its encoding
            rawdata = csv_file.read()
            result = chardet.detect(rawdata)
            encoding = result['encoding']
            
            # Reset the file pointer to the beginning of the file
            csv_file.seek(0)
            
            # If encoding detection fails, specify different encodings to try
            if encoding is None:
                encodings_to_try = ['utf-8', 'latin-1', 'ISO-8859-1']  # Add more encodings if needed
                for enc in encodings_to_try:
                    try:
                        # Attempt to decode the raw data using the current encoding
                        decoded_data = rawdata.decode(enc)
                        # Convert the decoded data to a StringIO object for pandas
                        csv_data = StringIO(decoded_data)
                        # Read the CSV file using pandas
                        df = pd.read_csv(csv_data)
                        break
                    except UnicodeDecodeError:
                        continue
            else:
                # Decode the raw data using the detected encoding
                decoded_data = rawdata.decode(encoding)
                # Convert the decoded data to a StringIO object for pandas
                csv_data = StringIO(decoded_data)
                # Read the CSV file using pandas
                df = pd.read_csv(csv_data)
            
            # Select the necessary columns
            selected_columns = df[['order_id', 'product_name', 'product_price', 'profit', 'quantity', 'category', 'sub_category', 'payment_mode', 'order_date', 'customer_name', 'state', 'city', 'gender', 'age']]
            
            # Perform top 3 analysis on selected columns
            top_3_product_price = top_3_analysis(selected_columns, 'product_price')
            top_3_profit = top_3_analysis(selected_columns, 'profit')
            top_3_quantity = top_3_analysis(selected_columns, 'quantity')
            
            # Convert the top 3 analysis results to HTML format for rendering in the template
            top_3_product_price_html = top_3_product_price.to_html(index=False)
            top_3_profit_html = top_3_profit.to_html(index=False)
            top_3_quantity_html = top_3_quantity.to_html(index=False)
            analyze_historical_sales(df)
            identify_seasonal_trends(df)
            calculate_profit_margins(df)
            max_sold_product_with_timeframes(df)
            # You can also return the top 3 analysis results to the template if needed
            return render(request, 'retailer/upload_csv.html', {'top_3_product_price_html': top_3_product_price_html, 'top_3_profit_html': top_3_profit_html, 'top_3_quantity_html': top_3_quantity_html})
        
        except Exception as e:
            # Handle any exceptions that may occur during file reading
            return render(request, 'retailer/upload_csv.html', {'error_message': str(e)})
    
    return render(request, 'retailer/upload_csv.html')


def top_3_analysis(df, column_name):
    # Get the top 3 rows based on the specified column
    top_3 = df.nlargest(3, column_name)
    return top_3



def analyze_historical_sales(df):
    # Calculate total sales
    total_sales = df['product_price'].sum()
    
    # Calculate average sales
    average_sales = df['product_price'].mean()
    
    # Calculate total profit
    total_profit = df['profit'].sum()
    
    # Calculate average profit margin
    average_profit_margin = (df['profit'] / df['product_price']).mean() * 100
    print(total_sales)
    print(total_profit)
    return {
        'total_sales': total_sales,
        'average_sales': average_sales,
        'total_profit': total_profit,
        'average_profit_margin': average_profit_margin
    }


def identify_seasonal_trends(df):
    # Assuming the 'order_date' column is in datetime format
        # Convert the 'order_date' column to datetime type if it's not already
    df['order_date'] = pd.to_datetime(df['order_date'], format='%d-%m-%Y')

    # Now you can use the .dt accessor on the 'order_date' column
    df['month'] = df['order_date'].dt.month
    # Group by month and calculate total sales for each month
    monthly_sales = df.groupby('month')['product_price'].sum()
    print(monthly_sales)
    return monthly_sales


def calculate_profit_margins(df):
    # Calculate profit margin for each product
    df['profit_margin'] = (df['profit'] / df['product_price']) * 100
    lol =df[['product_name', 'profit_margin']]
    print(lol)
    return df[['product_name', 'profit_margin']]


def max_sold_product_with_timeframes(df):
    # Assuming 'order_date' column is in datetime format
    df['year'] = df['order_date'].dt.year
    
    # Group by product and year, then calculate total quantity sold for each product in each year
    yearly_product_quantity = df.groupby(['product_name', 'year'])['quantity'].sum()
    
    # Find the max sold product and its corresponding year
    max_sold_product = yearly_product_quantity.idxmax()
    print(max_sold_product)
    return max_sold_product



# views.py
# views.py

import csv
from django.shortcuts import render
from .models import Order, Customer
# views.py

# views.py

import csv
from django.shortcuts import render, HttpResponse
from datetime import datetime
from .models import Order, Customer

def upload_csv_data(request):
    if request.method == 'POST':
        try:
            order_file = request.FILES['order_file']
            customer_file = request.FILES['customer_file']

            # Process order CSV file
            order_data = csv.DictReader(order_file.read().decode('utf-8').splitlines())
            for row in order_data:
                if all(key in row for key in ['Order ID', 'Amount', 'Profit', 'Quantity', 'Category', 'Sub-Category', 'PaymentMode']):
                    Order.objects.create(
                        order_id=row['Order ID'],
                        amount=row['Amount'],
                        profit=row['Profit'],
                        quantity=row['Quantity'],
                        category=row['Category'],
                        sub_category=row['Sub-Category'],
                        payment_mode=row['PaymentMode']
                    )
                else:
                    return HttpResponse('Error: Required keys are missing in the order CSV file.')

            # Process customer CSV file
            customer_data = csv.DictReader(customer_file.read().decode('utf-8').splitlines())
            for row in customer_data:
                if all(key in row for key in ['Order ID', 'Order Date', 'CustomerName', 'State', 'City']):
                    orders = Order.objects.filter(order_id=row['Order ID'])
                    if orders.exists():
                        order = orders.first()
                        try:
                            order_date = datetime.strptime(row['Order Date'], '%d-%m-%Y').strftime('%Y-%m-%d')
                        except ValueError:
                            return HttpResponse(f"Error: Invalid date format for Order ID {row['Order ID']}")
                        Customer.objects.create(
                            order=order,
                            order_date=order_date,
                            customer_name=row['CustomerName'],
                            state=row['State'],
                            city=row['City']
                        )
                    else:
                        return HttpResponse(f"Error: No order found with Order ID {row['Order ID']}")
                else:
                    return HttpResponse('Error: Required keys are missing in the customer CSV file.')

            return HttpResponse("Upload success")
        except Exception as e:
            return HttpResponse(f"Error: {str(e)}")
    return render(request, 'retailer/upload_data.html')



from django.db.models import Sum
from django.db.models.functions import TruncMonth

from .models import Order, Customer


from django.db.models import Sum
from django.shortcuts import render
from .models import Order

def popular_products(request):
    # Monthly Most Sold Products by Category and Sub-Category
    monthly_most_sold = Order.objects.values('category', 'sub_category').annotate(total_quantity=Sum('quantity')).order_by('-total_quantity')
    
    # Prime Shelf Space Allocation Category
    prime_category = Order.objects.values('category').annotate(total_quantity=Sum('quantity')).order_by('-total_quantity').first()

    monthly_most_sold_list = list(monthly_most_sold)
    # print(monthly_most_sold_list)

    # Retrieve top 3 most sold categories and their quantities
    top_categories = monthly_most_sold.values('category').annotate(total_quantity=Sum('quantity')).order_by('-total_quantity')[:3]

    # Initialize an empty list to store top 3 categories and their sub-categories
    top_categories_subcategories = []

    # Iterate over the top categories and fetch their corresponding sub-categories
    for category_data in top_categories:
        category = category_data['category']
        subcategories = monthly_most_sold.filter(category=category).order_by('-total_quantity')[:3]
        top_categories_subcategories.append({
            'category': category,
            'subcategories': subcategories
        })

    # Group by sub-category and calculate total profit
    subcategory_profit = Order.objects.values('sub_category').annotate(total_profit=Sum('profit')).order_by('-total_profit')

    # Initialize an empty list to store top sub-categories and their profits
    top_profit_subcategories = []

    # Iterate over the subcategories and fetch the top ones
    for subcategory_data in subcategory_profit[:6]:  # Change 3 to the number of top sub-categories you want to display
        top_profit_subcategories.append({
            'sub_category': subcategory_data['sub_category'],
            'total_profit': subcategory_data['total_profit']
        })

    print(prime_category)
    total_profits = Order.objects.aggregate(total_profits=Sum('profit'))

    # Calculate total sales
    total_sales = Order.objects.aggregate(total_sales=Sum('amount'))

    # Calculate total quantity sold
    total_quantity_sold = Order.objects.aggregate(total_quantity_sold=Sum('quantity'))


    total_amount_by_state = Customer.objects.values('state').annotate(total_amount=Sum('order__amount'))

    # Query to get total amount by customer name
    total_amount_by_customer = Customer.objects.values('customer_name').annotate(total_amount=Sum('order__amount'))

    # Query to get total profit by month
    total_profit_by_month = Order.objects.annotate(month=TruncMonth('customer__order_date')).values('month').annotate(total_profit=Sum('profit'))

    # Query to get total quantity by payment mode
    total_quantity_by_payment_mode = Order.objects.values('payment_mode').annotate(total_quantity=Sum('quantity'))

  
    print(total_amount_by_state)
    
    print(total_amount_by_customer)
    
    print(total_profit_by_month)
    
    print(total_quantity_by_payment_mode)
    context = {
            'top_categories_subcategories': top_categories_subcategories,
            'monthly_most_sold_list': monthly_most_sold_list,      
            'prime_category': prime_category,
            'top_profit_subcategories': top_profit_subcategories,
            'total_profits': total_profits['total_profits'] if total_profits['total_profits'] is not None else 0,
            'total_sales': total_sales['total_sales'] if total_sales['total_sales'] is not None else 0,
            'total_quantity_sold': total_quantity_sold['total_quantity_sold'] if total_quantity_sold['total_quantity_sold'] is not None else 0,
            'total_amount_by_state': total_amount_by_state,
            'total_amount_by_customer': total_amount_by_customer,
            'total_profit_by_month': total_profit_by_month,
            'total_quantity_by_payment_mode': total_quantity_by_payment_mode,
    
            }
    
    return render(request, 'retailer/popular_products.html', context)



# Sharvin

# views.py

from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import urllib.parse

load_dotenv()
genai.configure(api_key=(os.getenv("GOOGLE_API_KEY")))

def save_orders_from_excel(excel_file):
    df = pd.read_excel(excel_file)
    orders = []
    for _, row in df.iterrows():
        order = Order(
            order_id=row['order_id'],
            product_name=row['product_name'],
            product_price=row['product_price'],
            profit=row['profit'],
            quantity=row['quantity'],
            category=row['category'],
            sub_category=row['sub_category'],
            payment_mode=row['payment_mode'],
            order_date=row['order_date'],
            customer_name=row['customer_name'],
            state=row['state'],
            city=row['city'],
            gender=row['gender'],
            age=row['age']
        )
        orders.append(order)
    Order.objects.bulk_create(orders)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    print(text)
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(os.path.join(settings.BASE_DIR, "faiss_index"))
    return vector_store

def get_conversational_chain():
    prompt_template = """
    As a data analyst, you are tasked with performing a comprehensive analysis of a dataset containing information about product orders. The dataset includes columns such as 'order_id,' 'product_name,' 'product_price,' 'profit,' 'quantity,' 'category,' 'sub_category,' 'payment_mode,' 'order_date,' 'customer_name,' 'state,' 'city,' 'gender,' and 'age.' Your role is to derive valuable insights from this dataset and answer various questions regarding the products and sales.

Tasks to Perform:
1. Identify the most profitable product based on the 'profit' column.
2. Determine the product with the highest sales quantity.
3. Analyze the overall profit generated from all products.
4. Provide a breakdown of profits by category and sub-category.
5. Identify any patterns or trends in customer demographics (age, gender, location).

Answer each question thoroughly, utilizing statistical measures and insights. Present your analysis clearly and concisely, providing actionable information for stakeholders.

Ensure your response focuses on narrative explanations rather than including code snippets. Consider the entire dataset for accurate and insightful responses.

{context} (Provide the PDF containing the data for analysis)

Question:
{question}

Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local(os.path.join(settings.BASE_DIR, "faiss_index"), embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    response_text = response["output_text"]
    if response_text == "":
        response_text = "It seems that the answer is out of context. Here is a general response: ..."
    return response_text

def search_related_content(query):
    search_query = urllib.parse.quote(query)
    url = f"https://www.google.com/search?q={search_query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    search_results = soup.find_all('div', class_='BNeawe UPmit AP7Wnd')
    related_content = []
    for i, result in enumerate(search_results):
        if i >= 3:
            break
        related_content.append(result.text)
    return related_content

def scrape_youtube_videos(query):
    search_query = urllib.parse.quote(query)
    url = f"https://www.youtube.com/results?search_query={search_query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    video_results = soup.find_all('a', class_='yt-simple-endpoint style-scope ytd-video-renderer')
    related_videos = []
    for i, video in enumerate(video_results):
        if i >= 3:
            break
        video_title = video.get('title')
        video_link = f"https://www.youtube.com{video.get('href')}"
        related_videos.append((video_title, video_link))
    return related_videos

def display_related_content(related_content):
    return related_content

def gemini(request):
    if request.method == 'POST':
        # Handle PDF upload
        pdf_docs = request.FILES.getlist('pdf_files')
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

        # Handle user question
        user_question = request.POST.get('user_question')
        response_text = user_input(user_question)

        # Search related content
        related_content = search_related_content(user_question)
        youtube_content = scrape_youtube_videos(user_question)

        # Display related content
        related_content = display_related_content(related_content)

        # Return response
        return render(request, 'retailer/gemini.html', {'response_text': response_text, 'related_content': related_content})
    else:
        return render(request, 'retailer/gemini.html')


# def gemini(request):
#     if request.method == 'POST':
#         # Handle PDF upload
#         pdf_docs = request.FILES.getlist('pdf_files')
#         raw_text = get_pdf_text(pdf_docs)
#         text_chunks = get_text_chunks(raw_text)
#         get_vector_store(text_chunks)

#         # Handle user question
#         user_question = request.POST.get('user_question')
#         response_text = user_input(user_question)

#         # Search related content
#         related_content = search_related_content(user_question)
#         youtube_content = scrape_youtube_videos(user_question)

#         # Display related content
#         related_content = display_related_content(related_content)

#         # Return only text response
#         return render(request, 'retailer/gemini.html', {'response_text': response_text, 'related_content': related_content})
#     else:
#         return render(request, 'retailer/gemini.html')


# visualization/views.py
from .utils import generate_shelves

def visualize_shelves(request):
    data_file = 'mbd.csv'  # Adjust the path as per your file location
    shelves, frequent_itemsets_filtered_dict, top_selling_products = generate_shelves(data_file)  # Unpack the returned tuple
    frequent_itemsets_filtered = [(itemset, support) for itemset, support in frequent_itemsets_filtered_dict.items()]
    top_selling_products = [(product, quantity) for product, quantity in top_selling_products.items()]
    return render(request, 'retailer/visualize_shelves.html', {'shelves': shelves, 'frequent_itemsets': frequent_itemsets_filtered, 'top_selling_products': top_selling_products})

def get_top_and_least_sold_products_for_season(season, data):
    season_data = data[data['Season'] == season]
    top_products = season_data.groupby('Product').agg({'Quantity': 'sum'}).nlargest(3, 'Quantity')
    least_sold_products = season_data.groupby('Product').agg({'Quantity': 'sum'}).nsmallest(3, 'Quantity')
    return top_products, least_sold_products

def seasonal_analysis_view(request):
    # Read the CSV file
    data = pd.read_csv('Book2.csv')
    
    # Define seasons
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    
    # Get top and least sold products for each season
    seasonal_data = {}
    for season in seasons:
        top_products, least_sold_products = get_top_and_least_sold_products_for_season(season, data)
        seasonal_data[season] = {'top_products': top_products, 'least_sold_products': least_sold_products}
    print(seasonal_data)
    # Render the HTML template
    return render(request, 'retailer/season_analysis.html', {'seasonal_data': seasonal_data})

def threed_shelf(request):
    return render(request, 'retailer/3d.html')


# Sharvin

# views.py

from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import urllib.parse

load_dotenv()
genai.configure(api_key=(os.getenv("GOOGLE_API_KEY")))

def save_orders_from_excel(excel_file):
    df = pd.read_excel(excel_file)
    orders = []
    for _, row in df.iterrows():
        order = Order(
            order_id=row['order_id'],
            product_name=row['product_name'],
            product_price=row['product_price'],
            profit=row['profit'],
            quantity=row['quantity'],
            category=row['category'],
            sub_category=row['sub_category'],
            payment_mode=row['payment_mode'],
            order_date=row['order_date'],
            customer_name=row['customer_name'],
            state=row['state'],
            city=row['city'],
            gender=row['gender'],
            age=row['age']
        )
        orders.append(order)
    Order.objects.bulk_create(orders)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    print(text)
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(os.path.join(settings.BASE_DIR, "faiss_index"))
    return vector_store

def get_conversational_chain():
    prompt_template = """
    As a data analyst, you are tasked with performing a comprehensive analysis of a dataset containing information about product orders. The dataset includes columns such as 'order_id,' 'product_name,' 'product_price,' 'profit,' 'quantity,' 'category,' 'sub_category,' 'payment_mode,' 'order_date,' 'customer_name,' 'state,' 'city,' 'gender,' and 'age.' Your role is to derive valuable insights from this dataset and answer various questions regarding the products and sales.

Tasks to Perform:
1. Identify the most profitable product based on the 'profit' column.
2. Determine the product with the highest sales quantity.
3. Analyze the overall profit generated from all products.
4. Provide a breakdown of profits by category and sub-category.
5. Identify any patterns or trends in customer demographics (age, gender, location).

Answer each question thoroughly, utilizing statistical measures and insights. Present your analysis clearly and concisely, providing actionable information for stakeholders.

Ensure your response focuses on narrative explanations rather than including code snippets. Consider the entire dataset for accurate and insightful responses.

{context} (Provide the PDF containing the data for analysis)

Question:
{question}

Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local(os.path.join(settings.BASE_DIR, "faiss_index"), embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    response_text = response["output_text"]
    if response_text == "":
        response_text = "It seems that the answer is out of context. Here is a general response: ..."
    return response_text

def search_related_content(query):
    search_query = urllib.parse.quote(query)
    url = f"https://www.google.com/search?q={search_query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    search_results = soup.find_all('div', class_='BNeawe UPmit AP7Wnd')
    related_content = []
    for i, result in enumerate(search_results):
        if i >= 3:
            break
        related_content.append(result.text)
    return related_content

def scrape_youtube_videos(query):
    search_query = urllib.parse.quote(query)
    url = f"https://www.youtube.com/results?search_query={search_query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    video_results = soup.find_all('a', class_='yt-simple-endpoint style-scope ytd-video-renderer')
    related_videos = []
    for i, video in enumerate(video_results):
        if i >= 3:
            break
        video_title = video.get('title')
        video_link = f"https://www.youtube.com{video.get('href')}"
        related_videos.append((video_title, video_link))
    return related_videos

def display_related_content(related_content):
    return related_content

def gemini(request):
    if request.method == 'POST':
        # Handle PDF upload
        pdf_docs = request.FILES.getlist('pdf_files')
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

        # Handle user question
        user_question = request.POST.get('user_question')
        response_text = user_input(user_question)

        # Search related content
        related_content = search_related_content(user_question)
        youtube_content = scrape_youtube_videos(user_question)

        # Display related content
        related_content = display_related_content(related_content)

        # Return response
        return render(request, 'retailer/gemini.html', {'response_text': response_text, 'related_content': related_content})
    else:
        return render(request, 'retailer/gemini.html')


# def gemini(request):
#     if request.method == 'POST':
#         # Handle PDF upload
#         pdf_docs = request.FILES.getlist('pdf_files')
#         raw_text = get_pdf_text(pdf_docs)
#         text_chunks = get_text_chunks(raw_text)
#         get_vector_store(text_chunks)

#         # Handle user question
#         user_question = request.POST.get('user_question')
#         response_text = user_input(user_question)

#         # Search related content
#         related_content = search_related_content(user_question)
#         youtube_content = scrape_youtube_videos(user_question)

#         # Display related content
#         related_content = display_related_content(related_content)

#         # Return only text response
#         return render(request, 'retailer/gemini.html', {'response_text': response_text, 'related_content': related_content})
#     else:
#         return render(request, 'retailer/gemini.html')


# visualization/views.py
from .utils import generate_shelves

def visualize_shelves(request):
    data_file = 'mbd.csv'  # Adjust the path as per your file location
    shelves, frequent_itemsets_filtered_dict, top_selling_products = generate_shelves(data_file)  # Unpack the returned tuple
    frequent_itemsets_filtered = [([item for item in itemset], support) for itemset, support in frequent_itemsets_filtered_dict.items()]
    top_selling_products = [(product, quantity) for product, quantity in top_selling_products.items()]
    return render(request, 'retailer/visualize_shelves.html', {'shelves': shelves, 'frequent_itemsets': frequent_itemsets_filtered, 'top_selling_products': top_selling_products})

def get_top_and_least_sold_products_for_season(season, data):
    season_data = data[data['Season'] == season]
    top_products = season_data.groupby('Product').agg({'Quantity': 'sum'}).nlargest(3, 'Quantity')
    least_sold_products = season_data.groupby('Product').agg({'Quantity': 'sum'}).nsmallest(3, 'Quantity')
    return top_products, least_sold_products

def seasonal_analysis_view(request):
    data = pd.read_csv('Book2.csv')

    # Define seasons
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']

    # Get top and least sold products for each season and dynamically change shelves
    seasonal_data = {}
    for season in seasons:
        top_products, least_sold_products = get_top_and_least_sold_products_for_season(season, data)
        shelves = distribute_products_across_shelves(pd.Series(top_products), num_shelves=1, products_per_shelf=2)  # Adjusted to only one shelf
        seasonal_data[f'{season}, 1 Shelf'] = {'top_products': top_products, 'least_sold_products': least_sold_products, 'shelves': shelves}
    
    # Pass shelves data to template
    context = {'seasonal_data': seasonal_data}
    
    return render(request, 'retailer/season_analysis.html', context)

def threed_shelf(request):
    return render(request, 'retailer/3d.html')


from django.shortcuts import render
import pandas as pd

def get_top_and_least_sold_products_for_season(season, data):
    # Filter data for the given season
    season_data = data[data['Season'] == season]
    
    # Calculate sales volume for each product
    product_sales = season_data.groupby('Product')['Quantity'].sum().reset_index()
    
    # Sort products based on sales volume (descending order)
    sorted_products = product_sales.sort_values(by='Quantity', ascending=False)
    
    # Get top and least sold products
    top_products = sorted_products['Product'].tolist()
    least_sold_products = sorted_products.tail(1)['Product'].tolist()  # Assuming least sold product is the last one
    
    return top_products, least_sold_products

def distribute_products_across_shelves(popular_products, num_shelves, products_per_shelf):
    shelves = {}
    assigned_products = set()  # Track assigned products to avoid duplicates
    
    for i in range(num_shelves):
        start_idx = i * products_per_shelf
        end_idx = (i + 1) * products_per_shelf
        if i == num_shelves - 1:  # Last shelf may have fewer products
            end_idx = len(popular_products)
        shelf_products = []
        for product in popular_products[start_idx:end_idx]:
            if product not in assigned_products:
                shelf_products.append(product)
                assigned_products.add(product)
        shelves[f"Shelf {i+1}"] = shelf_products
    
    return shelves


# def total_amount_by_state():
#     """
#     Calculate total amount by state.
#     """
#     return 

# def total_amount_by_customer_name():
#     """
#     Calculate total amount by customer name.
#     """
#     return 

# def total_profit_by_month():
#     """
#     Calculate total profit by month.
#     """

# def total_quantity_by_payment_mode():
#     """
#     Calculate total quantity by payment mode.
#     """
#     return 


def index1(request):
    return render(request, 'index1.html')

from django.contrib.auth.decorators import login_required

def login(request):
    return render(request, 'pages/sign-in.html')


import os
from twilio.rest import Client

def send_report_via_sms(request):

    account_sid = os.environ['TWILIO_ACCOUNT_SID']
    auth_token = os.environ['TWILIO_AUTH_TOKEN']
    client = Client(account_sid, auth_token)

    message = client.messages.create(
                                from_='whatsapp:+14155238886',
                                body=f'Hello, Aryan \n Product Name : Bisleri Bottle is missing \n Shelf Location : D5 shelf \n Please refill and find the real-time video below',
                                to='whatsapp:+15005550006'
                            )

    print(message.sid)



def optimization(request):
    # Define the product data
    products = [
        {"name": "Iphone 15", "profit_priority": 1, "most_sold_priority": 1, "seasonal_priority": 1, "average_value": 1},
        {"name": "PlayStation", "profit_priority": 2, "most_sold_priority": 3, "seasonal_priority": 2, "average_value": 2},
        {"name": "Hoodies", "profit_priority": 4, "most_sold_priority": 2, "seasonal_priority": 3, "average_value": 3},
        {"name": "Floral Perfume", "profit_priority": 3, "most_sold_priority": 5, "seasonal_priority": 5, "average_value": 4},
        {"name": "Body Lotion", "profit_priority": 5, "most_sold_priority": 4, "seasonal_priority": 6, "average_value": 5},
        {"name": "Shower Gel", "profit_priority": 6, "most_sold_priority": 6, "seasonal_priority": 4, "average_value": 6},
        {"name": "Water Bottles", "profit_priority": 7, "most_sold_priority": 7, "seasonal_priority": 7, "average_value": 7},
        {"name": "Soap", "profit_priority": 9, "most_sold_priority": 8, "seasonal_priority": 8, "average_value": 8},
        {"name": "Hand Cream", "profit_priority": 8, "most_sold_priority": 9, "seasonal_priority": 9, "average_value": 9},
        {"name": "Serum", "profit_priority": 10, "most_sold_priority": 10, "seasonal_priority": 11, "average_value": 10},
        {"name": "Facewash", "profit_priority": 11, "most_sold_priority": 11, "seasonal_priority": 10, "average_value": 11},
        {"name": "Baidyanath Amla Juice", "profit_priority": 12, "most_sold_priority": 12, "seasonal_priority": 12, "average_value": 12},
       
    ]
    context = {
        'products' : products,
    }
    return render(request, 'retailer/optimization.html',context)



def optimized_products(request):
    return render(request, 'retailer/optimized_products.html')

import pandas as pd
from io import StringIO
import chardet

def upload(request):
    if request.method == 'POST' and 'csv_file' in request.FILES:   
        csv_file = request.FILES['csv_file']
        shelf_space = request.POST.get('shelf_space')
        print(shelf_space)
        
        # Check if the uploaded file is a CSV file
        if not csv_file.name.endswith('.csv'):
            return render(request, 'pages/upload.html', {'error_message': 'Please upload a CSV file.'})
        
        try:
            # Read the first few bytes of the file to detect its encoding
            rawdata = csv_file.read()
            result = chardet.detect(rawdata)
            encoding = result['encoding']
            
            # Reset the file pointer to the beginning of the file
            csv_file.seek(0)
            
            # If encoding detection fails, specify different encodings to try
            if encoding is None:
                encodings_to_try = ['utf-8', 'latin-1', 'ISO-8859-1']  # Add more encodings if needed
                for enc in encodings_to_try:
                    try:
                        # Attempt to decode the raw data using the current encoding
                        decoded_data = rawdata.decode(enc)
                        # Convert the decoded data to a StringIO object for pandas
                        csv_data = StringIO(decoded_data)
                        # Read the CSV file using pandas
                        df = pd.read_csv(csv_data)
                        break
                    except UnicodeDecodeError:
                        continue
            else:
                # Decode the raw data using the detected encoding
                decoded_data = rawdata.decode(encoding)
                # Convert the decoded data to a StringIO object for pandas
                csv_data = StringIO(decoded_data)
                # Read the CSV file using pandas
                df = pd.read_csv(csv_data)

                # Assuming 'Product' is the column name you want to extract
            products_list = df['Product'].unique().tolist()

            total_products = len(products_list)
            print(total_products)

            # Additional processing or storage logic can be added here

            return render(request, 'pages/upload.html', {'products_list': products_list, 'total_products': total_products})

        except Exception as e:
            return render(request, 'pages/upload.html', {'error_message': f'Error reading CSV file: {str(e)}'})

    return render(request, 'pages/upload.html')