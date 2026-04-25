import streamlit as st
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter



st.set_page_config(
    page_title="Cold Email RAG Search",
    page_icon="📧",
    layout="wide"
)


DOCUMENTS = [
    """
    Document 1: Lead Generation

    Lead generation is the process of identifying, attracting, and qualifying people or companies that may become customers. In business-to-business sales, a lead is usually a company or individual that fits a target market and may have a problem the business can solve. The purpose of lead generation is not simply to collect as many email addresses as possible, but to create a repeatable system for finding relevant prospects who have a realistic chance of becoming customers.

    There are two broad types of lead generation: inbound and outbound. Inbound lead generation happens when potential customers come to the company through channels such as search engines, content marketing, webinars, referrals, paid ads, or website forms. Outbound lead generation happens when the company proactively contacts potential customers through cold email, LinkedIn outreach, cold calls, partnerships, or direct messages. Both approaches can work well, but they require different strategies and success metrics.

    A strong lead generation process starts with defining the ideal customer profile. This includes identifying the industries, company sizes, job titles, regions, pain points, and buying triggers that make someone a good fit. After that, leads need to be sourced, cleaned, enriched, segmented, contacted, and measured. The most important metrics usually include reply rate, positive reply rate, meetings booked, opportunity rate, conversion rate, customer acquisition cost, and revenue generated. Good lead generation is therefore both a marketing activity and a data-driven sales process.
    """,

    """
    Document 2: Cold Email Outreach

    Cold email outreach is the practice of sending a business email to someone who has not previously interacted with the sender. It is commonly used in B2B sales, recruiting, partnerships, public relations, link building, and business development. Unlike spam, a good cold email is targeted, relevant, respectful, and written with a specific recipient or company type in mind. The goal is usually not to close a deal immediately, but to start a relevant conversation.

    Effective cold email outreach depends on three main elements: the quality of the lead list, the relevance of the message, and the deliverability of the sending domain. A strong message sent to the wrong audience will usually fail. A relevant lead list with a weak message will also underperform. Even if both the list and copy are strong, poor deliverability can prevent the email from reaching the inbox. This is why cold email should be treated as a system rather than just a writing task.

    A good cold email is usually short, specific, and easy to respond to. It should explain why the sender is reaching out, why the topic matters to the recipient, and what the next step should be. The most common performance metrics are delivery rate, open rate, reply rate, positive reply rate, bounce rate, unsubscribe rate, and booked meetings. The best outreach campaigns continuously improve by testing different segments, value propositions, subject lines, and follow-up sequences.
    """,

    """
    Document 3: Ideal Customer Profile

    An ideal customer profile, often shortened to ICP, describes the type of company or person that is most likely to benefit from a product or service. In B2B sales, an ICP usually includes firmographic details such as industry, company size, location, revenue, funding stage, business model, technology used, and growth signals. At the contact level, it may include job title, seniority, department, responsibilities, decision-making authority, and likely pain points.

    Defining the ICP is one of the most important steps in cold email because it determines who receives the message. If the ICP is too broad, the campaign may reach many people who do not care about the offer. If the ICP is too narrow, the campaign may not generate enough volume. A useful ICP balances relevance and scale. For example, instead of targeting “all SaaS companies,” a better ICP might be “B2B SaaS companies with 50–300 employees that recently raised funding and are hiring sales development representatives.”

    A strong ICP should be based on evidence rather than assumptions. Companies can analyze their best current customers, highest-converting past campaigns, sales call notes, CRM data, website analytics, and customer interviews. The ICP should also change over time. If one industry replies more often, books more meetings, or converts into paying customers at a higher rate, that segment should receive more focus. In this way, the ICP becomes a living part of the sales strategy.
    """,

    """
    Document 4: Cold Email Personalization

    Personalization in cold email means adapting the message to the specific recipient, company, or situation. Basic personalization includes using a first name, company name, job title, or industry. More meaningful personalization references a business event, recent hiring activity, funding announcement, technology stack, content post, company initiative, or likely pain point. The purpose is to show that the sender has a real reason for reaching out.

    Good personalization is not just a compliment or random observation. It should connect directly to the reason for the email. For example, mentioning that a company is hiring many account executives is useful if the sender helps sales teams improve prospecting, onboarding, enablement, or CRM workflows. Mentioning a podcast appearance or LinkedIn post may be less useful unless it naturally connects to the value proposition. The best personalization creates relevance quickly.

    Personalization should also be efficient. A cold email does not need a long custom paragraph. Often one sentence is enough if it is specific and relevant. Over-personalization can make the email feel unnatural, especially if the sender references something too personal or unrelated. The best approach is to combine segmentation with light personalization: group similar leads by shared pain points, then add one specific detail that explains why the message is relevant to that recipient.
    """,

    """
    Document 5: Subject Lines

    A cold email subject line is the short text that appears in the recipient's inbox before they open the email. Its main purpose is to create enough curiosity or relevance for the recipient to open the message. A good subject line should be clear, short, and connected to the content of the email. It should not trick the recipient, exaggerate the offer, or look like mass marketing.

    Strong subject lines often use simple language. Examples include “quick question,” “idea for company,” “sales pipeline,” “reducing manual prospecting,” or “company hiring.” These subject lines work because they are easy to understand and do not feel overly promotional. In many B2B contexts, a plain and conversational subject line performs better than a clever or heavily branded one.

    Bad subject lines often use spam-like words, excessive punctuation, fake urgency, or misleading claims. Phrases such as “guaranteed results,” “limited time offer,” “free money,” or “open immediately” can hurt trust and deliverability. The subject line should also match the body of the email. If the subject promises one topic and the email discusses something else, recipients may ignore future messages or mark the email as spam. Testing subject lines is useful, but the biggest rule is simple: be relevant and honest.
    """,

    """
    Document 6: Cold Email Structure

    A strong cold email usually follows a simple structure: relevant opening, problem or context, value proposition, proof, and call to action. The recipient should quickly understand why they are being contacted and why the message matters. Because most professionals receive many emails, cold emails should be easy to scan and should avoid unnecessary background information.

    The opening line should establish relevance. This could be a reference to the recipient's role, company situation, industry trend, hiring activity, or business challenge. The problem statement should describe an issue the recipient is likely to recognize. The value proposition should then explain how the sender helps solve that problem. Strong value propositions focus on outcomes such as saving time, increasing qualified meetings, improving deliverability, reducing manual work, or increasing conversion rates.

    Proof makes the message more credible. This can be a short customer example, a result, a recognizable client type, or a specific method. The call to action should be simple and low-friction. Instead of asking for a long meeting immediately, many effective emails ask whether the topic is relevant, whether the recipient is the right person, or whether they are open to a short conversation. The best cold emails feel like the start of a useful conversation, not a full sales pitch.
    """,

    """
    Document 7: Follow-Up Strategy

    Follow-up emails are an important part of cold outreach because many recipients do not respond to the first message. They may be busy, traveling, unsure if the topic is relevant, or simply miss the email. A well-designed sequence usually includes three to five follow-ups spaced over several days or weeks. The purpose is to stay visible without becoming annoying.

    A good follow-up should add value or introduce a new angle. Instead of repeating the same message, the sender can share a short case study, ask a different question, mention a relevant pain point, provide a useful insight, or clarify the offer. For example, if the first email focused on saving time, a follow-up might focus on increasing reply rates or reducing manual prospecting. This gives the recipient more reasons to engage.

    Timing and tone matter. Sending follow-ups too frequently can feel aggressive and increase unsubscribes or spam complaints. Waiting too long can cause the recipient to forget the context. A common pattern is to wait two or three business days after the first email, then gradually increase the delay between later follow-ups. The final follow-up, sometimes called a breakup email, should be polite and give the recipient an easy way to say no or redirect the sender to the right person.
    """,

    """
    Document 8: Email Deliverability

    Email deliverability is the ability of an email to successfully reach the recipient's inbox rather than being blocked, bounced, or sent to spam. Deliverability is one of the most important parts of cold email because even the best copy will fail if the email never reaches the recipient. Deliverability depends on technical setup, sender reputation, list quality, sending behavior, and recipient engagement.

    Important technical foundations include SPF, DKIM, and DMARC records. These authentication settings help mailbox providers verify that the sender is allowed to send emails from the domain. Sender reputation is affected by bounce rates, spam complaints, sending volume, domain age, reply rates, and how recipients interact with messages. A new sending domain should usually be warmed up gradually rather than used for high-volume outreach immediately.

    Good deliverability practices include verifying email addresses before sending, avoiding purchased low-quality lists, keeping bounce rates low, writing simple non-spammy messages, avoiding too many links or images, and making it easy for recipients to opt out. Positive engagement, such as replies, helps sender reputation. Poor engagement, high bounce rates, and spam complaints can damage deliverability across an entire domain.
    """,

    """
    Document 9: Measuring Cold Email Performance

    Cold email performance should be measured with several metrics because no single metric tells the full story. Delivery rate shows whether emails are reaching recipients. Open rate can indicate whether subject lines and sender names are working, but it is less reliable because of privacy protections and image-blocking technologies. Reply rate shows whether recipients are engaging with the message. Positive reply rate is often more important because not every reply is valuable.

    Other important metrics include bounce rate, unsubscribe rate, spam complaint rate, meeting booked rate, opportunity creation rate, and conversion rate. A campaign with a high open rate but low reply rate may have a subject line that creates curiosity but body copy that fails to create interest. A campaign with many replies but few positive replies may be reaching the wrong audience or using a message that creates confusion. A campaign with strong reply rates but few meetings may have a weak call to action or poor follow-up process.

    The best cold email analysis connects email metrics to business outcomes. A campaign should not be judged only by opens or replies. It should be judged by whether it creates qualified conversations, opportunities, customers, or revenue. This is why sales teams often track the funnel from sent emails to delivered emails, opened emails, replies, positive replies, booked meetings, sales opportunities, and closed deals.
    """,

    """
    Document 10: Lead Scoring

    Lead scoring is the process of assigning a value or score to a lead based on how likely they are to become a customer. A lead score can be based on demographic fit, firmographic fit, behavioral engagement, and sales activity. For example, a lead from the right industry, with the right job title, who visited several important website pages and replied positively to an email would receive a higher score than a lead with little engagement and poor fit.

    Lead scoring helps sales teams prioritize their time. Instead of treating every lead equally, the team can focus first on leads with the highest probability of conversion. This is especially useful when a business has more leads than the sales team can manually handle. A good lead scoring system can improve efficiency, reduce wasted outreach, and increase the chance that high-intent prospects receive attention quickly.

    Lead scoring can be rule-based or model-based. A rule-based system might add points for specific actions, such as opening an email, clicking a link, visiting the pricing page, or matching the ideal customer profile. A model-based system uses historical data to learn which features predict conversion. Machine learning and deep learning models can improve lead scoring by identifying patterns that are difficult to capture manually, but they must be carefully checked for data leakage and bias.
    """
]


@st.cache_resource
def build_vector_db():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_text("\n\n".join(DOCUMENTS))

    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="cold_email_documents"
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        ids=ids
    )

    return collection, chunks

collection, chunks = build_vector_db()


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Search", "About"])


if page == "Home":
    st.title("📧 Cold Email RAG Search App")

    st.write("""
    This app is a searchable knowledge base about lead generation and cold email outreach.
    It uses semantic search to find relevant information from a collection of documents.
    """)

    st.subheader("What this app can help with")
    st.write("""
    - Understanding cold email strategy
    - Learning about lead generation
    - Finding information about deliverability
    - Exploring lead scoring and segmentation
    - Reviewing follow-up and copywriting best practices
    """)

    st.subheader("Dataset")
    st.write(f"Number of source documents: {len(DOCUMENTS)}")
    st.write(f"Number of text chunks after splitting: {len(chunks)}")

    st.subheader("Example questions")
    st.write("""
    - What is lead generation?
    - How can I improve cold email deliverability?
    - What makes a good cold email subject line?
    - Why is personalization important?
    - What is lead scoring?
    """)


elif page == "Search":
    st.title("🔍 Semantic Search")

    query = st.text_input("Ask a question about cold email or lead generation:")

    number_of_results = st.slider(
        "Number of results",
        min_value=1,
        max_value=5,
        value=3
    )

    if query:
        results = collection.query(
            query_texts=[query],
            n_results=number_of_results
        )

        st.subheader("Search Results")

        for i, document in enumerate(results["documents"][0], start=1):
            st.markdown(f"### Result {i}")
            st.write(document)
            st.divider()


elif page == "About":
    st.title("About this App")

    st.write("""
    This application was built as a Retrieval-Augmented Generation assignment.
    It uses Streamlit for the web interface, LangChain for text processing,
    and ChromaDB as the vector database.
    """)

    st.subheader("Chunking Strategy")
    st.write("""
    The documents are split using a chunk size of 500 characters and an overlap of 100 characters.
    This was chosen because the documents are educational explanations. A larger chunk gives enough
    context for each search result, while the overlap helps prevent important ideas from being cut off.
    """)

    st.subheader("Embedding / Search Method")
    st.write("""
    The app uses ChromaDB's built-in embedding and similarity search functionality.
    This allows the app to find relevant results based on meaning rather than only exact keyword matching.
    """)