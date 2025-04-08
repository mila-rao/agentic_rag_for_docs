from typing import Dict, Any, Optional
import logging
import json
from crewai import Agent, Task, Crew, Process
from crewai.tasks.task_output import TaskOutput

from retrieval.hybrid import HybridRetriever
from crewai import LLM

logger = logging.getLogger(__name__)


class RAGCrew:
    """Implements an agentic RAG system using CrewAI."""

    def __init__(
            self,
            retriever: HybridRetriever,
            llm_api_key: str,
            model_name: str = "openai/o3-mini",
            verbose: bool = False
    ):
        """Initialize the RAG Crew.

        Args:
            retriever: HybridRetriever instance for document retrieval
            llm_api_key: OpenAI API key for the LLM
            model_name: Model name to use (default: o3-mini)
            verbose: Whether to enable verbose logging
        """
        self.retriever = retriever
        self.llm_api_key = llm_api_key
        self.model_name = model_name
        self.verbose = verbose

        # Initialize the LLM
        self.llm = LLM(
            api_key=llm_api_key,
            temperature=0.2,
            model=model_name
        )

        logger.info(f"Initialized RAG Crew with model: {model_name}")

    def setup_agents(self):
        """Create and configure the agent crew.

        Returns:
            Tuple of (agents, tasks)
        """
        # 1. Query Planner Agent
        planner = Agent(
            role="Query Planner",
            goal="Analyze user queries to plan the retrieval strategy.",
            backstory="""You are an expert at understanding user intent and breaking down 
            complex questions into searchable components. You can identify key concepts,
            entities, and relationships in queries.""",
            verbose=self.verbose,
            llm=self.llm,
            allow_delegation=True
        )

        # 2. Information Retriever Agent
        retriever_agent = Agent(
            role="Information Retriever",
            goal="Find the most relevant and accurate information from the knowledge base.",
            backstory="""You are a master at finding information. You know how to craft
            effective search queries and can judge the relevance of retrieved documents.
            You're thorough and always look for multiple perspectives on a topic.""",
            verbose=self.verbose,
            llm=self.llm,
            allow_delegation=True
        )

        # 3. Information Synthesizer Agent
        synthesizer = Agent(
            role="Information Synthesizer",
            goal="Create coherent, accurate responses by combining retrieved information.",
            backstory="""You excel at understanding context and synthesizing information
            from multiple sources. You can identify connections between different pieces
            of information and organize them into a coherent narrative that directly
            addresses the user's query.""",
            verbose=self.verbose,
            llm=self.llm,
            allow_delegation=True
        )

        # 4. Self-Critic Agent
        critic = Agent(
            role="Information Critic",
            goal="Ensure accuracy, completeness, and clarity of the final response.",
            backstory="""You are meticulous about fact-checking and ensuring logical
            consistency. You can identify gaps, contradictions, or unsubstantiated claims
            in information. You ensure that all assertions are properly supported by the
            retrieved documents.""",
            verbose=self.verbose,
            llm=self.llm,
            allow_delegation=True
        )

        # Define Tasks
        # 1. Query Planning Task
        planning_task = Task(
            description="""
            Analyze the user query to:
            1. Identify if it's a simple keyword search or a complex question
            2. Identify key concepts, entities, and relationships
            3. Break down complex questions into sub-questions if needed
            4. Suggest relevant filters or constraints (e.g., date ranges, document types)

            Output your analysis as a structured plan including:
            - Query type (keyword/question)
            - Main intent
            - Key search terms
            - Sub-questions (if applicable)
            - Suggested filters
            """,
            agent=planner,
            expected_output="""
            A structured query plan in JSON format with the following fields:
            {
                "query_type": "keyword" or "question",
                "main_intent": "brief description of main intent",
                "key_terms": ["list", "of", "key", "terms"],
                "sub_questions": ["question1", "question2"] or [],
                "filters": {"field": "value"} or {}
            }
            """
        )

        # 2. Information Retrieval Task
        retrieval_task = Task(
            description="""
            Based on the query plan, retrieve the most relevant information:
            1. For each key term or sub-question, perform a search using the retrieval system
            2. For keyword searches, prioritize exact matches
            3. For questions, prioritize contextual relevance
            4. Apply any filters from the query plan
            5. Ensure diversity of sources
            6. Judge the relevance and quality of each retrieved chunk

            Use the following function to retrieve information:
            self.retrieve_documents(query, filters, top_k)

            Output the retrieved information with relevance assessments.
            """,
            agent=retriever_agent,
            expected_output="""
            Retrieved information with source details and relevance assessments in JSON format:
            {
                "retrieved_chunks": [
                    {
                        "text": "chunk content",
                        "source": "document source",
                        "relevance": "high/medium/low",
                        "rationale": "why this is relevant"
                    }
                ]
            }
            """
        )

        # 3. Information Synthesis Task
        synthesis_task = Task(
            description="""
            Using the retrieved information:
            1. Integrate information from all relevant sources
            2. Organize information logically to address the query
            3. Identify and resolve conflicts or contradictions
            4. Ensure all assertions are supported by the retrieved information
            5. Maintain proper attribution to sources
            6. Use clear, concise language appropriate for the user's query

            Output a comprehensive response that directly addresses the user's query.
            """,
            agent=synthesizer,
            expected_output="""
            A comprehensive answer with the following structure:
            {
                "answer": "The complete synthesized answer",
                "sources": ["list of sources used"],
                "confidence": "high/medium/low",
                "gaps": "any identified information gaps"
            }
            """
        )

        # 4. Self-Critique Task
        critique_task = Task(
            description="""
            Critically evaluate the synthesized response:
            1. Verify all factual claims against the retrieved information
            2. Check for logical consistency and coherence
            3. Identify any unsupported assertions
            4. Ensure the response directly addresses the user's query
            5. Check for completeness and clarity
            6. Suggest improvements or corrections

            Output your critique and a revised response if needed.
            """,
            agent=critic,
            expected_output="""
            A critique and final response in JSON format:
            {
                "critique": {
                    "factual_accuracy": "assessment",
                    "logical_consistency": "assessment",
                    "completeness": "assessment",
                    "clarity": "assessment",
                    "issues": ["list of issues"] or []
                },
                "final_answer": "The revised answer",
                "sources": ["list of sources used"]
            }
            """
        )

        agents = [planner, retriever_agent, synthesizer, critic]
        tasks = [planning_task, retrieval_task, synthesis_task, critique_task]

        return agents, tasks

    def process_query(self, query: str, filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a user query through the agent crew.

        Args:
            query: The user's query
            filter_dict: Optional metadata filter

        Returns:
            Dict containing the final response and supporting information
        """
        # Fast path for simple keyword searches
        if len(query.split()) <= 3 and self.retriever.is_keyword_search(query):
            logger.info(f"Using fast path for keyword search: {query}")
            return self._process_keyword_search(query, filter_dict)

        # Set up the agents and tasks
        agents, tasks = self.setup_agents()

        # Create the crew
        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=self.verbose,
            process=Process.sequential  # Use sequential processing for RAG tasks
        )

        # Prepare task inputs
        inputs = {
            "query": query,
            "filter_dict": json.dumps(filter_dict) if filter_dict else "{}"
        }

        # Add retrieval method to the retriever agent
        setattr(agents[1], "retrieve_documents", self._agent_retrieve_documents)

        # Execute the crew
        result = crew.kickoff(inputs=inputs)

        # Process final result
        try:
            final_output = self._process_crew_result(result, query)
            return final_output
        except Exception as e:
            logger.error(f"Error processing crew result: {str(e)}")
            return {
                "answer": f"Error processing your query: {str(e)}",
                "sources": [],
                "success": False
            }

    def _agent_retrieve_documents(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 5) -> str:
        """Document retrieval method for agents to use.

        Args:
            query: The search query
            filters: Optional metadata filters
            top_k: Number of results to return

        Returns:
            JSON string of retrieved documents
        """
        # Parse filters if provided as string
        if isinstance(filters, str):
            try:
                filters = json.loads(filters)
            except:
                filters = None

        # Retrieve documents
        texts, metadatas, scores = self.retriever.retrieve(query, filters, top_k)

        # Format results
        results = []
        for text, metadata, score in zip(texts, metadatas, scores):
            result = {
                "text": text,
                "source": metadata.get("source", "Unknown"),
                "score": float(score),
                "metadata": metadata
            }
            results.append(result)

        # Return as JSON string
        return json.dumps({"results": results})

    def _process_keyword_search(self, query: str, filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a simple keyword search query.

        Args:
            query: The keyword search query
            filter_dict: Optional metadata filter

        Returns:
            Dict containing search results
        """
        # Retrieve documents
        texts, metadatas, scores = self.retriever.retrieve(
            query,
            filter_dict,
            top_k=10,
            force_mode="keyword"
        )

        # Format results
        results = []
        for text, metadata, score in zip(texts, metadatas, scores):
            result = {
                "text": text,
                "source": metadata.get("source", "Unknown"),
                "score": float(score),
                "metadata": metadata
            }
            results.append(result)

        return {
            "type": "search_results",
            "query": query,
            "results": results,
            "success": True
        }

    def _process_crew_result(self, result: TaskOutput, query: str) -> Dict[str, Any]:
        """Process the final output from the crew.

        Args:
            result: The TaskOutput from the crew
            query: The original query

        Returns:
            Dict containing the final response
        """
        # Extract the final critic's output
        final_output = result.raw

        # Try to parse as JSON if possible
        try:
            output_dict = json.loads(final_output)

            # Extract the final answer and sources
            final_answer = output_dict.get("final_answer", "")
            sources = output_dict.get("sources", [])

            if not final_answer and "critique" in output_dict:
                # Check if answer is nested in critique structure
                final_answer = output_dict.get("critique", {}).get("final_answer", "")

            return {
                "type": "question_answer",
                "query": query,
                "answer": final_answer,
                "sources": sources,
                "full_critique": output_dict.get("critique", {}),
                "success": True
            }

        except json.JSONDecodeError:
            # If not valid JSON, return the raw text
            logger.warning("Failed to parse crew result as JSON")

            return {
                "type": "question_answer",
                "query": query,
                "answer": final_output,
                "sources": [],
                "success": True
            }