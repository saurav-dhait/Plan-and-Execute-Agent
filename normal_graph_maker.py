from langchain import hub
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import operator
from typing import Annotated, List, Tuple, TypedDict
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import Union
from langgraph.graph import StateGraph, START
from typing import Literal
import functools

load_dotenv()


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
                    "If you need to further use tools to get the answer, use Plan."
    )


def get_agent_executor(llm):
    tools = [TavilySearchResults(max_results=3)]
    prompt = hub.pull("wfh/react-agent-executor")
    prompt.pretty_print()
    agent_executor = create_react_agent(llm, tools, messages_modifier=prompt)
    return agent_executor


def get_planner(llm):
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """For the given objective, come up with a simple step by step plan. \
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
            ),
            ("placeholder", "{messages}"),
        ]
    )
    return planner_prompt | llm.with_structured_output(Plan)


def get_replanner(llm):
    replanner_prompt = ChatPromptTemplate.from_template(
        """For the given objective, come up with a simple step by step plan. \
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

    Your objective was this:
    {input}

    Your original plan was this:
    {plan}

    You have currently done the following steps:
    {past_steps}

    Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
    )
    return replanner_prompt | llm.with_structured_output(Act)


def execute_step(state: PlanExecute, agent_executor):
    plan = state["plan"]
    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
    {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = agent_executor.invoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content), ]
    }


def plan_step(state: PlanExecute, planner):
    plan = planner.invoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


def replan_step(state: PlanExecute, replanner):
    output = replanner.invoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
    if "response" in state and state["response"]:
        return "__end__"
    else:
        return "agent"


def get_graph():
    # some global declarations
    llm = ChatGroq(
        temperature=0,
        model="llama3-70b-8192",
    )
    agent_executor = get_agent_executor(llm)
    planner = get_planner(llm)
    replanner = get_replanner(llm)
    workflow = StateGraph(PlanExecute)

    # make partial functions for every node function
    plan_step_partial = functools.partial(plan_step, planner=planner)
    execute_step_partial = functools.partial(execute_step, agent_executor=agent_executor)
    replan_step_partial = functools.partial(replan_step, replanner=replanner)

    # Add the plan node
    workflow.add_node("planner", plan_step_partial)
    # Add the execution step
    workflow.add_node("agent", execute_step_partial)
    # Add a replan node
    workflow.add_node("replan", replan_step_partial)
    workflow.add_edge(START, "planner")
    # From the plan we go to agent
    workflow.add_edge("planner", "agent")
    # From agent, we replan
    workflow.add_edge("agent", "replan")
    workflow.add_conditional_edges(
        "replan",
        # Next, we pass in the function that will determine which node is called next.
        should_end,
    )
    # Finally we compile and return the graph as a LangChain Runnable,
    return workflow.compile()

# config = {"recursion_limit": 50}
# inputs = {"input": "who is lemans race winner 2024 ?"}
#
# graph = get_graph()
# for event in graph.stream(inputs, config=config):
#     for k, v in event.items():
#         if k != "__end__":
#             print(v)
