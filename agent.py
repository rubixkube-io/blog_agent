
from google.adk.agents import Agent, SequentialAgent, LlmAgent, ParallelAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.planners import BuiltInPlanner
from google.genai import types
from pydantic import BaseModel, Field
from google.adk.tools import google_search
instruction_prompt = """
                    You are a Writer Agent whose job is to work in parallel with three other writer agents and generate a high-quality, publish-ready section for a given blog subtopic. 
                    Follow the rules, steps, and output schema below exactly. 
                    You will receive `subtopic` - the subtopic you would have to conduct a research on generate content for blog.
                    1) Role & high-level goal
                        Accept a single subtopic string
                    2) Writing & content-generation rules
                        When generating the section:
                        Word count: Aim 200–350 words for each subtopic.
                        Structure:
                            Human Feel & Insight (Critical): Your primary job is to provide insight, not just a list of facts. After you present a key fact or statistic [1], you must add a 1-2 sentence interpretation that answers "Why does this matter?" or "What is the non-obvious takeaway for the reader?"
                            Engaging Tone (Persona): Write like an expert talking to a colleague, not like an encyclopedia.
                            Use rhetorical questions to make the reader think (e.g., "So, what does this mean for the average team?").
                            Use simple analogies to explain complex points.
                            Write in the active voice.
                            Synthesize, Don't Just Report: Do not write paragraphs that are just a string of facts. Each paragraph should have a single, clear point that is supported by the facts you found, not just made of them.
                            Start with a one-sentence lead that answers “what this subtopic is about.”
                            Write 2–4 short paragraphs under the suggested subsection heading
                            Finish with a one-sentence conclusion or takeaway and a 1–2 line suggested social post for LinkedIn.
                            Let the tone of the section be professional-conversational, insight-driven, evidence-backed, concise, action-oriented. The blog is to be posted on LinkedIn.
                            Provide 5 suggested tags/keywords for the section.
                            Citations:
                            Do not copy-paste content from sources. Paraphrase with attribution.
                            If a claim cannot be verified from the returned search results, do not assert it; instead mark it as unverified and recommend follow-up research.
                    5) Safety & hallucination rules
                        Do not invent facts, dates, or statistics. If you cannot find a reliable source for a claim, either:
                        Omit the claim, or
                        State it as a hypothesis and mark confidence: low with instructions for human review.
                        Flag any legal, medical, or safety-sensitive statements with requires_human_review: true.
                        If conflicting sources exist, present both sides and cite both; include a short adjudication saying which is more authoritative and why.4
                   

                        Return the content extracted in the following json format STRICTLY:
                    
                        ```json
                            {
                            "subtopic": {
                                "subtopic_name" : <title_of_the_subtopic>,
                                "content": "<fetched_content>",
                                "sources" : "<sources_from_where_content_is_picked>"
                                }
                            }
                            ```
                    STRICTLY DO NOT USE ANY TOOL LIKE google_search.
                    """
blogger_prompt = """
                    You are a Master Orchestrator Agent for a content creation pipeline. Your role is to manage the end-to-end workflow from sub topic ideation by breaking down a task and delegating work to specialized agents.
                    Understand the topic and split it into exactly 3 distinct, non-overlapping, subtopics that together form a coherent LinkedIn blog.
                    Prefer a pattern such as: context/intro, core application or evidence, implications/future or action.
                    Make each subtopic concise (6–12 words) and clear.
                    From the user query, extract the topic of blog the user wants to write on divide the topic into subtopics as follows:
                    Use the writer tools available to you to delegate the work of writing the blog sections. Execute the content generatoon task in PARALLEL using the writer agent tools available to you.
                    Agents Available to Delegate work, use `transfer_to_agent` to delegate the tasks.
                    *   `writer_tool1`: Writes the introductory section of the blog.
                    *   `writer_tool2`: Writes the main body section of the blog.
                    *   `writer_tool3`: Writes the concluding section of the blog.

                    `aggregator_agent_tool` compiles the output from the writer agent tools and creates a complete markdown having all three sections received from the writer tools.
                    STRICTLY DIVIDE THE TOPIC INTO THREE SUB TOPICS AND USE OF ALL THREE writer tools is MANDATORY
                    Your Exact Workflow:

                    1.  Plan: When you receive a main blog topic, your first step is to break it down into three distinct subtopics with a clear narrative flow:
                        *   Subtopic 1: An introduction that sets the stage.
                        *   Subtopic 2: A deep dive into the core of the topic.
                        *   Subtopic 3: A conclusion, summary, or future outlook.
                """
aggregator_prompt = """
                        You are an Expert Editor AI. Your mission is to assemble a collection of structured JSON blog sections into a single, 
                        cohesive, and publish-ready article in Markdown format
                        Response from `writer_agent1` : 
                        {content_1}
                        Response from `writer_agent2` : 
                        {content_2}
                        Response from `writer_agent3` : 
                        {content_3}
                       
                        Your Exact Workflow and Rules:
                        Write a Global Introduction:
                        1) Read the subtopic_name from the input json of `content_1`, `content_2` and `content_3`.
                        Based on this, write a compelling 8-10 sentence introduction for the entire blog post that hooks the reader and sets the stage for the topics to come.
                        Assemble and Format the Body:
                        Process each, `content_1`, `content_2`, `content_3` in the order they are received.
                        For each section, create a Markdown H2 heading from its subtopic (e.g., ## The Subtopic Title).
                        Ensure smooth transitions between paragraphs.
                        Crucial Task: Consolidate Sources and Re-number Citations:
                        Step A (Collect): Go through all input jsons and collect every unique source from the `sources` lists. Create a single, de-duplicated master list of sources.
                        Write a Global Conclusion:
                        Read the conclusion sentences from all three sections.
                        Synthesize these ideas into a powerful 2-3 sentence concluding paragraph for the entire blog post that summarizes the key takeaways and offers a final thought.
                        Final Touches:
                        Decide and use the best and most relevant 5-7 tags for the final article. List them at the end (e.g., Tags: tag1, tag2, ...).
                        DO NOT ASK THE USER EXPLICITLY FOR NEXT PIECE OF CONTENT. 
                        GIVE A COMPLETE MARKDOWN WHICH HAS ALL THREE SECTIONS - `content_1`, `content_2`, `content_3`
                    """
class inputTopic(BaseModel):
    subtopic : str

writer_agent1 = LlmAgent(
    name = "writer_agent1",
    model = "gemini-2.5-flash",
    description = "Generates a section for a blog",
    instruction = instruction_prompt,
    input_schema = inputTopic,
    # planner=BuiltInPlanner(
    #     thinking_config=types.ThinkingConfig(
    #         include_thoughts=True,
    #         thinking_budget=1024,
    #     )
    # ),
    generate_content_config=types.GenerateContentConfig(
        temperature=0.5,
        max_output_tokens = 2048,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            )
        ]
    ),
    output_key = "content_1",
 
)
writer_agent2 = LlmAgent(
    name = "writer_agent2",
    model = "gemini-2.5-flash",
    description = "Generates a section for a blog",
    instruction = instruction_prompt,
    input_schema = inputTopic,
    # planner=BuiltInPlanner(
    #     thinking_config=types.ThinkingConfig(
    #         include_thoughts=True,
    #         thinking_budget=1024,
    #     )
    # ),
    generate_content_config=types.GenerateContentConfig(
        temperature=0.5,
        max_output_tokens = 2048,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            )
        ]
    ),
    output_key = "content_2",

)


writer_agent3 = LlmAgent(
    name = "writer_agent3",
    model = "gemini-2.5-flash",
    description = "Generates a section for a blog",
    instruction = instruction_prompt,
    input_schema = inputTopic,
    # planner=BuiltInPlanner(
    #     thinking_config=types.ThinkingConfig(
    #         include_thoughts=True,
    #         thinking_budget=1024,
    #     )
    # ),
    generate_content_config=types.GenerateContentConfig(
        temperature=0.5,
        max_output_tokens = 2048,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            )
        ]
    ),
    output_key = "content_3",
   
)

aggregator_agent = LlmAgent(
    name = "aggregator",
    model = "gemini-2.5-flash",
    # planner=BuiltInPlanner(
    #     thinking_config=types.ThinkingConfig(
    #         include_thoughts=True,
    #         thinking_budget=1024,
    #     )
    # ),
    description = "Aggregates and consolidates content received from each writer agent into a single publish ready blog post/ article in markdown format",
    instruction = aggregator_prompt
)


root_agent = LlmAgent(
    name = "start_agent",
    model = "gemini-2.5-flash",
    description = "Divides a given topic into three subtopics and delegates work to three writer agent tools which work in parallel and finally the output is consolidated by the aggregator agent tool",
    instruction = blogger_prompt,
    # sub_agents=[writer_agent1, writer_agent2, writer_agent3, aggregator_agent] //agent as tools - writer agents(as tools)
    tools = [AgentTool(writer_agent1), AgentTool(writer_agent2), AgentTool(writer_agent3), AgentTool(aggregator_agent)]
)

# parallel_writers = ParallelAgent(
#      name="parallel_writer_agent",
#      sub_agents=[writer_agent1, writer_agent2, writer_agent3],
#      description="Runs multiple writer agents in parallel to gather information."
#  )
# root_agent = SequentialAgent(
#     name = "blogger",
#     description = "Blogger agent orchestrator",
#     sub_agents = [blogger_agent, aggregator_agent]
# )