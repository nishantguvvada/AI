from langchain.prompts import PromptTemplate
from langchain.llms import VertexAI
from langchain.chains import TransformChain, SequentialChain, LLMChain
import pandas_gbq
import pandas as pd
import base64
from io import BytesIO
import plotly.figure_factory as ff
import gradio as gr

pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Here is where we kick things off generating a query from the user's question.
primary_template = """Question:
You are an expert BigQuery user.  You have access to the genie_metrics table.  
The genie_metrics table is in the `rax-datascience-dev` project in the `sandbox` dataset.  The
genie_metrics table contains bookings and pipeline information for Rackspace.  The DATE_TIME column contains
the date of the metric value in the bigquery date format.  The METRIC column contains a string of which type 
of metric value is in the row (BOOKINGS or PIPELINE).  The VALUE column contains the metric value
as a numeric.  Please note that the DATE_TIME column actually contains full dates and not just Year and Month
values.  For this reason, any time the query requires aggregation by month or year, you should use the
FORMAT_DATE function appropriately.  Remember that in BigQuery, the format for year is %Y and not YYYY.
Additionally, the format for month is %m and not MM.  Remeber that the order of operators for the
FORMAT_DATE function is FORMAT_DATE(format,date), for example FORMAT_DATE('%Y-%m',DATE_TIME).
Remeber when aggregating queries you should use the appropiate alias in the GROUP BY clause.
Given this information, please construct a query that answers the following question:
{question}
The query should be valid BigQuery Standard SQL syntax and should execute properly when run.  It should
not contain any extra characters or syntax that is not necessary to answer the question.  The query should
not have any extra backticks or sql identifiers.  
Answer: """
primary_prompt = PromptTemplate(
        template=primary_template,
    input_variables=['question']
)

llm = VertexAI(
   max_tokens=10000,
)
primary_llm_chain = LLMChain(prompt=primary_prompt, llm=llm, output_key="query")



# The primary LLM call generates a query.  This converts that query into a csv string of data.
def query2data(inputs: dict) -> dict:
   # take in the query and return data as a csv string
    query = inputs["query"]
    print(query)
    df = pandas_gbq.read_gbq(query)
    print("dtypes1",df.dtypes)
    return_obj = {'df':df, 'csv_string':df.to_csv()}
    return return_obj

csv_chain = TransformChain(
    input_variables=["query"], output_variables=["df","csv_string"], transform=query2data
)

# This transform takes in a csv string and returns a plotly figure.
def data2plot(inputs: dict) -> dict:
    df = inputs["df"]
    #df.convert_dtypes()
    print("dtypes2",df.dtypes)
    ax = df.plot.bar(x=0,y=1)
    ax.ticklabel_format(style='plain', axis='y')
    fig = ax.get_figure()
    fig.tight_layout()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded_fig = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    tbl =  ff.create_table(df)
    tmpfile2 = BytesIO()
    tbl.write_image(tmpfile2)
    encoded_tbl = base64.b64encode(tmpfile2.getvalue()).decode('utf-8')
    html = '<table><tr><td style="vertical-align:top"><img src=\'data:image/png;base64,{}\'</td><td style="vertical-align:top"><img src=\'data:image/png;base64,{}\'></td></tr></table>'.format(encoded_fig, encoded_tbl)
    return {"output_text": html}

plot_chain = TransformChain(
    input_variables=["df"], output_variables=["output_text"], transform=data2plot
)

# Here we'll take the original question and the df and ask the LLM to summarize the data.
secondary_template = """Question:
You are a data analyst.  You have access to the following data:
{csv_string}
Given this information, please answer the following question:
{question}
Use a business-like tone in your answer.  The answer should be a summary of the data and should not
contain any extra characters or syntax that is not necessary to answer the question.  The answer should
be returned as a formatted html paragraph with the overall summary of the data in text format.  
A good example answer would be <p>The total damage of storms for 2010 was $1,000,000. The largest
month was January with $500,000 in damage.  The smallest month was February with $100,000 in damage.</p>
Answer: 
"""

secondary_prompt = PromptTemplate(
    template=secondary_template,
    input_variables=['question','csv_string']
)
secondary_llm_chain = LLMChain(prompt=secondary_prompt, llm=llm, output_key="summary")

def formatter(inputs: dict) -> dict:
   # take in the query and return data as a csv string
    query = inputs["query"]
    summary = inputs["summary"]
    graphs = inputs["output_text"]
    output_html = "<h2>Summary</h2><p>{}</p><p>{}</p><h2>Additional Resources</h2><p>Query:<pre>{}</pre></p>".format(summary,graphs,query)
    return {'output_html':output_html}

transformer_chain = TransformChain(
    input_variables=["query","output_text","summary"], output_variables=["output_html"], transform=formatter
)


sequential_chain = SequentialChain(
   chains=[primary_llm_chain, csv_chain, plot_chain, secondary_llm_chain, transformer_chain],
   input_variables=["question"],
   output_variables=[
      "output_html",
      ]
)

# user question
question = """What was total bookings by month for 2022?"""

def bq_graph_chain(question):
    output = sequential_chain.run(question)
    return output

with gr.Blocks() as demo:
    gr.Markdown(
    ## Graph demo
"""
    Ask the Genie any question. The Genie will generate a BigQuery query that answers your question and will render a graphical version of your answer.
    (Note: This is demo and only connected to a single table with high-level metrics.) 
"""
    )
    with gr.Row():
      with gr.Column():
        input_text = gr.Textbox(label="Question", value=question, )
    with gr.Row():
      generate = gr.Button("Ask Genie")
    with gr.Row():
      label2 = gr.HTML("Answer will appear here")
    generate.click(fn=bq_graph_chain, inputs=input_text, outputs=[label2],api_name="bq_graph_chain")
demo.launch(share=False, debug=False)