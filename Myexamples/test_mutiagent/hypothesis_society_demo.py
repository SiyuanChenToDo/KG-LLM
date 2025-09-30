import sys
print(f"Python Version: {sys.version}")
print(f"Sys Path: {sys.path}")

# NOTE: You can set the CAMEL_API_KEY env variable in .env file or as shown below.
# os.environ["CAMEL_API_KEY"] = "<YOUR_API_KEY>"

# NOTE: You can set the OPENAI_API_KEY env variable in .env file or as shown below.
# os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

# NOTE: You can set the ANTHROPIC_API_KEY env variable in .env file or as shown below.
# os.environ["ANTHROPIC_API_KEY"] = "<YOUR_API_KEY>"

# NOTE: You can set the MISTRAL_API_KEY env variable in .env file or as shown below.
# os.environ["MISTRAL_API_KEY"] = "<YOUR_API_KEY>"

import textwrap
import os
from getpass import getpass
from typing import Dict, Any, Optional, List
from datetime import datetime
import re
import json

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.models.qwen_model import QwenModel
from camel.tasks import Task
from camel.toolkits import FunctionTool, SearchToolkit
from camel.types import ModelPlatformType, ModelType
from camel.societies.workforce import Workforce
from Myexamples.agents.graph_agents import get_scholar_scour_config, get_qwen_leader_config, get_qwen_editor_config, get_idea_igniter_config, run_local_rag


class OutputFormatter:
    """Professional output formatting with color support for enhanced readability"""
    
    # ANSI color codes
    COLORS = {
        'red': '\033[91m',
        'green': '\033[92m', 
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'bold': '\033[1m',
        'underline': '\033[4m',
        'reset': '\033[0m'
    }
    
    @classmethod
    def _colorize(cls, text: str, color: str) -> str:
        """Apply color to text if terminal supports it"""
        if sys.stdout.isatty():  # Only colorize if output is to terminal
            return f"{cls.COLORS.get(color, '')}{text}{cls.COLORS['reset']}"
        return text
    
    @classmethod
    def success(cls, message: str):
        """Print success message in green"""
        print(cls._colorize(f"[SUCCESS] {message}", 'green'))
    
    @classmethod
    def error(cls, message: str):
        """Print error message in red"""
        print(cls._colorize(f"[ERROR] {message}", 'red'))
    
    @classmethod
    def warning(cls, message: str):
        """Print warning message in yellow"""
        print(cls._colorize(f"[WARNING] {message}", 'yellow'))
    
    @classmethod
    def info(cls, message: str):
        """Print info message in blue"""
        print(cls._colorize(f"[INFO] {message}", 'blue'))
    
    @classmethod
    def header(cls, message: str):
        """Print header message in bold"""
        print(cls._colorize(f"\n{message}", 'bold'))
    
    @classmethod
    def section(cls, title: str, width: int = 80):
        """Print section separator"""
        separator = "=" * width
        print(f"\n{cls._colorize(separator, 'cyan')}")
        print(cls._colorize(f"{title.center(width)}", 'bold'))
        print(cls._colorize(separator, 'cyan'))


class HypothesisGenerationSociety:
    """
    A streamlined collaborative CAMEL society using 8 specialized Qwen agents
    for generating novel scientific hypotheses through multi-agent collaboration.
    
    Agents:
    - Dr. Qwen Leader: Chief researcher and synthesis expert (handles preliminary and final synthesis)
    - Scholar Scour: Literature review expert with RAG integration
    - Idea Igniter: Advanced creative innovation specialist (deep literature analysis, cross-domain thinking)
    - Dr. Qwen Technical: Technical rigor specialist  
    - Dr. Qwen Practical: Applied research specialist
    - Prof. Qwen Ethics: Impact and significance analyst
    - Critic Crucible: Peer review specialist
    - Prof. Qwen Editor: Scientific writing and style specialist
    """

    def __init__(self):
        self.workforce = None
        self.agent_configs = {}  # Store agent configurations
        # API keys are now set up in the initial cell
        OutputFormatter.success("Scientific Hypothesis Generation Society initialized successfully")

    def create_qwen_agent(
        self, role_name: str, system_prompt: str = None, persona: str = None, 
        specialization: str = None, model_type: str = None, tools: Optional[List[Any]] = None) -> ChatAgent:
        """Create a Qwen agent with specific role and complete system prompt"""

        # Record agent configuration
        config = {
            "role_name": role_name,
            "model_type": model_type or "plus",
            "specialization": specialization or "General research",
            "prompt_length": len(system_prompt) if system_prompt else len(persona or "") + 200
        }
        self.agent_configs[role_name] = config

        # If a complete system prompt is provided, use it directly
        if system_prompt:
            msg_content = textwrap.dedent(system_prompt).strip()
        else:
            # Fallback to the old method for backward compatibility
            msg_content = textwrap.dedent(f"""
            You are {role_name}, a distinguished researcher in the scientific community.

            Your persona: {persona}

            Your specialization: {specialization}

            You are part of an elite collaborative research team dedicated to generating novel, 
            testable scientific hypotheses that advance human knowledge. Your team follows 
            rigorous scientific methodology and interdisciplinary thinking.

            When collaborating:
            1. Provide detailed, evidence-based analysis within your expertise
            2. Build constructively upon other team members' contributions
            3. Maintain the highest standards of scientific rigor and intellectual honesty
            4. Think creatively while respecting established scientific principles
            5. Always support your reasoning with logical arguments and evidence
            6. Consider both theoretical foundations and practical implications
            7. Embrace interdisciplinary perspectives and cross-domain insights
            """).strip()

        sys_msg = BaseMessage.make_assistant_message(
            role_name=role_name,
            content=msg_content,
        )

        # Configure Qwen model with enhanced settings
        if model_type == "max":
            qwen_model_type = ModelType.QWEN_MAX
        elif model_type == "plus":
            qwen_model_type = ModelType.QWEN_PLUS
        elif model_type == "turbo":
            qwen_model_type = ModelType.QWEN_TURBO
        else:
            qwen_model_type = ModelType.QWEN_PLUS  # 默认使用 QWEN_PLUS

        model = QwenModel(
            model_type=qwen_model_type,
            api_key=os.getenv("QWEN_API_KEY"),
            url=os.getenv("QWEN_API_BASE_URL"),  # 可选，如果不设置将使用默认值
        )
        
        # Handle SearchToolkit specifically
        processed_tools = []
        if tools:
            for tool_item in tools:
                if isinstance(tool_item, SearchToolkit):
                    processed_tools.extend(tool_item.get_tools())
                else:
                    processed_tools.append(tool_item)

        return ChatAgent(
            system_message=sys_msg,
            model=model,
            tools=processed_tools,
        )

    def display_agent_configs(self):
        """Display all agent configurations before running"""
        OutputFormatter.section("SCIENTIFIC HYPOTHESIS GENERATION TEAM CONFIGURATION")
        
        for i, (agent_name, config) in enumerate(self.agent_configs.items(), 1):
            print(f"\n{i}. {agent_name}")
            print(f"   Model: Qwen {config['model_type'].upper()}")
            print(f"   Role: {config['specialization']}")
            print(f"   Prompt Length: {config['prompt_length']} characters")
        
        print(f"\nTotal Agents: {len(self.agent_configs)}")
        model_distribution = {}
        for config in self.agent_configs.values():
            model_type = config['model_type']
            model_distribution[model_type] = model_distribution.get(model_type, 0) + 1
        
        print("\nModel Distribution:")
        for model, count in model_distribution.items():
            print(f"   - Qwen {model.upper()}: {count} agents")
        
        print("=" * 80)

    def create_research_workforce(self):
        """Create the collaborative hypothesis generation workforce using Qwen models"""
        OutputFormatter.info("Creating Scientific Hypothesis Generation Society with Qwen models")

        # Create Qwen agents using encapsulated configurations
        leader_conf = get_qwen_leader_config()
        qwen_lead = self.create_qwen_agent(
            role_name=leader_conf["role_name"],
            system_prompt=leader_conf["system_prompt"],
            model_type=leader_conf["model_type"],
            tools=leader_conf["tools"],
        )


        prof_qwen_ethics_prompt = """
        You are Prof. Qwen Ethics, an expert in science policy and impact.

        ## Your Task:
        Analyze the provided creative ideas for their potential significance and broader impact. Your output MUST be a single, valid JSON object.

        ## JSON OutputFormat:
        {{
            "impact_analyses": [
                {{
                    "idea_summary": "<A one-sentence summary of the creative idea you are analyzing>",
                    "significance_score": <A numerical score from 1-5 on the potential scientific significance>,
                    "significance_reasoning": "<A brief justification for the significance score>",
                    "impact_score": <A numerical score from 1-5 on the potential broader (societal, technological) impact>,
                    "impact_reasoning": "<A brief justification for the impact score>"
                }}
            ]
        }}

        ## Instructions:
        - Create one JSON object inside the "impact_analyses" list for each creative idea provided.
        - Do NOT output anything other than the single JSON object.
        """
        
        qwen_ethicist = self.create_qwen_agent(
            role_name="Prof. Qwen Ethics",
            system_prompt=prof_qwen_ethics_prompt,
            model_type="max")
        # Note: Do NOT set output_language - CAMEL Workforce handles JSON wrapping

        # Create more Qwen agents for technical and practical perspectives
        dr_qwen_technical_prompt = """
        You are Dr. Qwen Technical, an expert in theoretical sciences.

        ## Your Task:
        Analyze the provided creative ideas for technical and logical soundness. Your output MUST be a single, valid JSON object.

        ## JSON Output Format:
        {{
            "technical_analyses": [
                {{
                    "idea_summary": "<A one-sentence summary of the creative idea you are analyzing>",
                    "plausibility_score": <A numerical score from 1-5 on the mechanistic plausibility>,
                    "plausibility_reasoning": "<A brief justification for the plausibility score>",
                    "consistency_score": <A numerical score from 1-5 on the logical consistency>,
                    "consistency_reasoning": "<A brief justification for the consistency score, noting any hidden assumptions or flaws>"
                }}
            ]
        }}

        ## Instructions:
        - Create one JSON object inside the "technical_analyses" list for each creative idea provided.
        - Do NOT output anything other than the single JSON object.
        """
        
        qwen_technical = self.create_qwen_agent(
            role_name="Dr. Qwen Technical",
            system_prompt=dr_qwen_technical_prompt,
            model_type="max")
        # Note: Do NOT set output_language - CAMEL Workforce handles JSON wrapping

        dr_qwen_practical_prompt = """
        You are Dr. Qwen Practical, an expert in experimental science.

        ## Your Task:
        Analyze the provided creative ideas for experimental testability. Your output MUST be a single, valid JSON object.

        ## JSON Output Format:
        {{
            "practical_analyses": [
                {{
                    "idea_summary": "<A one-sentence summary of the creative idea you are analyzing>",
                    "falsifiability_score": <A numerical score from 1-5 on how specific and falsifiable the idea is>,
                    "falsifiability_reasoning": "<A brief justification for the falsifiability score>",
                    "feasibility_score": <A numerical score from 1-5 on the practical feasibility of testing the idea>,
                    "feasibility_reasoning": "<A brief justification for the feasibility score>",
                    "suggested_approach": "<A brief description of the most direct experimental approach (e.g., RCT, simulation)>"
                }}
            ]
        }}

        ## Instructions:
        - Create one JSON object inside the "practical_analyses" list for each creative idea provided.
        - Do NOT output anything other than the single JSON object.
        """
        
        qwen_practical = self.create_qwen_agent(
            role_name="Dr. Qwen Practical",
            system_prompt=dr_qwen_practical_prompt,
            model_type="max")
        # Note: Do NOT set output_language - CAMEL Workforce handles JSON wrapping

        # Create new specialized agents for hypothesis generation (extracted module)
        scholar_conf = get_scholar_scour_config()
        scholar_scour = self.create_qwen_agent(
            role_name=scholar_conf["role_name"],
            system_prompt=scholar_conf["system_prompt"],
            model_type=scholar_conf["model_type"],
            tools=scholar_conf["tools"],
        )

        # Create Idea Igniter agent using extracted configuration
        igniter_conf = get_idea_igniter_config()
        idea_igniter = self.create_qwen_agent(
            role_name=igniter_conf["role_name"],
            system_prompt=igniter_conf["system_prompt"],
            model_type=igniter_conf["model_type"],
            tools=igniter_conf["tools"],
        )

        critic_crucible_prompt = """
        You are Critic Crucible, an expert peer reviewer.

        ## Your Task:
        Critically review the preliminary hypothesis draft. You MUST provide your output in a valid JSON format.

        ## JSON Output Format:
        {{
            "quality_score": <A numerical score from 1 to 5. A score below 4 indicates a need for significant revision>,
            "strengths": "<A brief paragraph summarizing the most compelling aspects of the hypothesis.>",
            "weaknesses": "<A brief paragraph summarizing the most critical flaws or areas needing improvement.>",
            "recommendations": [
                "<A specific, actionable suggestion for improvement.>",
                "<Another specific, actionable suggestion.>",
                "<... more suggestions ...>"
            ]
        }}

        ## Instructions:
        - Your review must be objective and constructive.
        - The `quality_score` should reflect the overall scientific merit, clarity, and testability of the hypothesis.
        - If the score is below 4, the `recommendations` MUST be clear and detailed enough to guide a revision.
        - Do NOT output anything other than the JSON object.
        """
        
        critic_crucible = self.create_qwen_agent(
            role_name="Critic Crucible",
            system_prompt=critic_crucible_prompt,
            model_type="max")
        # Note: Do NOT set output_language - CAMEL Workforce handles JSON wrapping

        # Create a new agent for final polishing
        editor_prompt = """
        You are Prof. Qwen Editor, an expert scientific editor with a keen eye for clarity, flow, and professional presentation.

        ## Your Task:
        Review the provided scientific hypothesis report for style, grammar, and structure. You are NOT evaluating the scientific merit, but the quality of the writing. Your output MUST be a valid JSON object.

        ## JSON Output Format:
        {{
            "clarity_score": <A numerical score from 1.0 to 10.0 for readability and clarity>,
            "consistency_score": <A numerical score from 1.0 to 10.0 for consistency in terminology and voice>,
            "recommendations": [
                {{
                    "type": "<'STYLE'|'GRAMMAR'|'STRUCTURE'|'REDUNDANCY'>",
                    "suggestion": "<A specific, actionable suggestion for improving the text.>"
                }}
            ]
        }}

        ## Instructions:
        - Focus solely on the quality of the writing.
        - Be pedantic about grammar and punctuation.
        - Identify any awkward phrasing or redundant sentences.
        - Check if the report flows logically from one section to the next.
        - Ensure a consistent, professional tone throughout the document.
        - Do NOT output anything other than the JSON object.
        """
        editor_conf = get_qwen_editor_config()
        qwen_editor = self.create_qwen_agent(
            role_name=editor_conf["role_name"],
            system_prompt=editor_conf["system_prompt"],
            model_type=editor_conf["model_type"],
            tools=editor_conf["tools"],
        )
        # Note: Do NOT set output_language - CAMEL Workforce handles JSON wrapping

        # Create coordinator and task agents with Qwen models
        coordinator_model = QwenModel(
            model_type=ModelType.QWEN_MAX,
            api_key=os.getenv("QWEN_API_KEY"),
            url=os.getenv("QWEN_API_BASE_URL"),
        )
        
        task_model = QwenModel(
            model_type=ModelType.QWEN_MAX,
            api_key=os.getenv("QWEN_API_KEY"),
            url=os.getenv("QWEN_API_BASE_URL"),
        )

        # Create ChatAgent instances for coordinator and task planning
        coordinator_agent = ChatAgent(
            model=coordinator_model,
        )
        
        task_agent = ChatAgent(
            model=task_model,
        )

        # Create the workforce with custom coordinator and task agents
        # Increased timeout to 2400s (40 min) for enhanced Idea Igniter with detailed output
        self.workforce = Workforce(
            'Scientific Hypothesis Generation Society (Qwen)',
            coordinator_agent=coordinator_agent,
            task_agent=task_agent,
            task_timeout_seconds=2400.0,  # 40 minutes for complex LLM tasks with long prompts
        )
        OutputFormatter.success("Coordinator and task agents configured with extended timeout.")

        # Add all 6 agents with descriptive roles
        self.workforce.add_single_agent_worker(
            'Dr. Qwen Leader (Chief Researcher & Synthesis Expert)',
            worker=qwen_lead,
        ).add_single_agent_worker(
            'Scholar Scour (Literature Analysis Expert)',
            worker=scholar_scour,
        ).add_single_agent_worker(
            'Idea Igniter (Creative Innovation Specialist)',
            worker=idea_igniter,
        ).add_single_agent_worker(
            'Dr. Qwen Technical (Technical Rigor Specialist)',
            worker=qwen_technical,
        ).add_single_agent_worker(
            'Dr. Qwen Practical (Applied Research Specialist)',
            worker=qwen_practical,
        ).add_single_agent_worker(
            'Prof. Qwen Ethics (Impact & Significance Analyst)',
            worker=qwen_ethicist,
        ).add_single_agent_worker(
            'Critic Crucible (Peer Review Specialist)',
            worker=critic_crucible,
        ).add_single_agent_worker(
            'Prof. Qwen Editor (Scientific Writing Specialist)',
            worker=qwen_editor)

        OutputFormatter.success(
            "Scientific Hypothesis Generation Society created with 8 "
            "specialized agents")
        return self.workforce

    def _extract_final_synthesis(self, raw_output: str) -> str:
        """
        Extracts the FINAL synthesis output from Dr. Qwen Leader for use in iteration loops.
        This is used for passing to the next iteration, not for final report generation.
        """
        # Split by subtask markers
        parts = re.split(r'---\s*Subtask\s+[\w\-\.]+\s+Result\s*---', raw_output)
        
        if len(parts) <= 1:
            return raw_output.strip()
        
        # The LAST part is Dr. Qwen Leader's final output
        final_output = parts[-1].strip()
        final_output = re.sub(r'\.\.\.$', '', final_output).strip()
        
        # Fallback logic for failed tasks
        if len(final_output) < 100 or "Task processing failed" in final_output:
            for part in reversed(parts[:-1]):
                cleaned = part.strip()
                if len(cleaned) > 200 and "## Executive Summary" in cleaned:
                    return cleaned
        
        return final_output
    
    def _structure_final_report(self, raw_output: str) -> str:
        """
        将原始输出重组为结构化的、有条理的最终报告。
        保留所有中间过程（可解释性），但以清晰的层级结构组织。
        
        报告结构：
        1. 执行摘要（来自 Dr. Qwen Leader 的最终综合）
        2. 文献综述（来自 Scholar Scour）
        3. 研究想法（来自 Idea Igniter）
        4. 技术分析（来自分析 agents 的 JSON）
        5. 最终科学假设（来自 Dr. Qwen Leader 的完整综合）
        6. 生成过程追踪（原始 subtask 标记，作为附录）
        """
        # 解析所有 subtask 结果
        subtask_pattern = r'---\s*Subtask\s+([\w\-\.]+)\s+Result\s*---'
        parts = re.split(subtask_pattern, raw_output)
        
        # 构建 subtask 映射: {task_id: content}
        subtasks = {}
        for i in range(1, len(parts), 2):
            if i < len(parts) - 1:
                task_id = parts[i]
                content = parts[i + 1].strip()
                subtasks[task_id] = content
        
        if not subtasks:
            # 如果没有 subtask 标记，返回原始内容
            return raw_output
        
        # 识别各个部分（假设任务 ID 格式为 hypothesis_generation_TIMESTAMP.X）
        task_ids = sorted(subtasks.keys())
        
        literature_review = ""
        ideas = ""
        technical_analysis = ""
        practical_analysis = ""
        impact_analysis = ""
        final_synthesis = ""
        
        for task_id in task_ids:
            content = subtasks[task_id]
            
            # 根据内容特征识别每个部分
            if "## Literature Review" in content or "### Established Knowledge" in content:
                literature_review = content
            elif "### Idea" in content and "**1. Core Mechanism" in content:
                ideas = content
            elif '"technical_analyses"' in content or '"plausibility_score"' in content:
                technical_analysis = self._format_json_analysis(content, "Technical Analysis")
            elif '"practical_analyses"' in content or '"feasibility_score"' in content:
                practical_analysis = self._format_json_analysis(content, "Practical Analysis")
            elif '"impact_analyses"' in content or '"significance_score"' in content:
                impact_analysis = self._format_json_analysis(content, "Impact Analysis")
            elif "## Executive Summary" in content or ("## Background and Rationale" in content and "## Detailed Hypothesis" in content):
                final_synthesis = content
        
        # 组装结构化报告
        structured_report = []
        
        # Part 1: 执行摘要（提取自最终综合）
        if final_synthesis:
            exec_summary = self._extract_section(final_synthesis, "## Executive Summary", "## Background")
            if exec_summary:
                structured_report.append("# Executive Summary\n\n" + exec_summary)
        
        # Part 2: 文献综述
        if literature_review:
            structured_report.append("---\n\n# 1. Literature Review\n\n" + literature_review)
        
        # Part 3: 创意研究想法
        if ideas:
            structured_report.append("---\n\n# 2. Novel Research Ideas\n\n" + ideas)
        
        # Part 4: 多维度分析
        if technical_analysis or practical_analysis or impact_analysis:
            structured_report.append("---\n\n# 3. Multi-Dimensional Analysis\n")
            if technical_analysis:
                structured_report.append("\n## 3.1 Technical Rigor Analysis\n\n" + technical_analysis)
            if practical_analysis:
                structured_report.append("\n## 3.2 Practical Feasibility Analysis\n\n" + practical_analysis)
            if impact_analysis:
                structured_report.append("\n## 3.3 Impact & Significance Analysis\n\n" + impact_analysis)
        
        # Part 5: 最终科学假设
        if final_synthesis:
            structured_report.append("---\n\n# 4. Final Scientific Hypothesis\n\n" + final_synthesis)
        
        # Part 6: 生成过程追踪（作为附录）
        structured_report.append("\n\n---\n\n# Appendix: Generation Process Trace\n\n")
        structured_report.append("<details>\n<summary>Click to expand: Raw generation process with subtask markers</summary>\n\n")
        structured_report.append("```\n" + raw_output[:5000] + "\n...\n```\n")
        structured_report.append("</details>")
        
        return "\n\n".join(structured_report)
    
    def _extract_section(self, text: str, start_marker: str, end_marker: str) -> str:
        """从文本中提取特定部分"""
        start_idx = text.find(start_marker)
        if start_idx == -1:
            return ""
        
        end_idx = text.find(end_marker, start_idx + len(start_marker))
        if end_idx == -1:
            # 如果没有结束标记，取到下一个 ## 标题或文本结尾
            next_section = text.find("\n## ", start_idx + len(start_marker))
            if next_section != -1:
                end_idx = next_section
            else:
                return text[start_idx + len(start_marker):].strip()
        
        return text[start_idx + len(start_marker):end_idx].strip()
    
    def _format_json_analysis(self, json_content: str, title: str) -> str:
        """将 JSON 分析格式化为可读的 markdown"""
        try:
            # 尝试解析 JSON
            data = json.loads(json_content.strip())
            
            formatted = []
            
            # 根据分析类型提取数据
            if "technical_analyses" in data:
                analyses = data["technical_analyses"]
                for i, analysis in enumerate(analyses, 1):
                    formatted.append(f"### Idea {i}\n")
                    formatted.append(f"**Summary**: {analysis.get('idea_summary', 'N/A')}\n")
                    formatted.append(f"- **Plausibility Score**: {analysis.get('plausibility_score', 'N/A')}/5")
                    formatted.append(f"  - *Reasoning*: {analysis.get('plausibility_reasoning', 'N/A')}")
                    formatted.append(f"- **Consistency Score**: {analysis.get('consistency_score', 'N/A')}/5")
                    formatted.append(f"  - *Reasoning*: {analysis.get('consistency_reasoning', 'N/A')}\n")
            
            elif "practical_analyses" in data:
                analyses = data["practical_analyses"]
                for i, analysis in enumerate(analyses, 1):
                    formatted.append(f"### Idea {i}\n")
                    formatted.append(f"**Summary**: {analysis.get('idea_summary', 'N/A')}\n")
                    formatted.append(f"- **Falsifiability Score**: {analysis.get('falsifiability_score', 'N/A')}/5")
                    formatted.append(f"  - *Reasoning*: {analysis.get('falsifiability_reasoning', 'N/A')}")
                    formatted.append(f"- **Feasibility Score**: {analysis.get('feasibility_score', 'N/A')}/5")
                    formatted.append(f"  - *Reasoning*: {analysis.get('feasibility_reasoning', 'N/A')}")
                    if 'suggested_approach' in analysis:
                        formatted.append(f"- **Suggested Approach**: {analysis['suggested_approach']}\n")
            
            elif "impact_analyses" in data:
                analyses = data["impact_analyses"]
                for i, analysis in enumerate(analyses, 1):
                    formatted.append(f"### Idea {i}\n")
                    formatted.append(f"**Summary**: {analysis.get('idea_summary', 'N/A')}\n")
                    formatted.append(f"- **Significance Score**: {analysis.get('significance_score', 'N/A')}/5")
                    formatted.append(f"  - *Reasoning*: {analysis.get('significance_reasoning', 'N/A')}")
                    formatted.append(f"- **Broader Impact Score**: {analysis.get('impact_score', 'N/A')}/5")
                    formatted.append(f"  - *Reasoning*: {analysis.get('impact_reasoning', 'N/A')}\n")
            
            return "\n".join(formatted) if formatted else json_content
            
        except (json.JSONDecodeError, KeyError, TypeError):
            # 如果解析失败，返回原始内容（可能已经是格式化的）
            return json_content
    
    def _extract_and_parse_json(self, raw_output: str) -> list[Dict[str, Any]]:
        """
        Extracts all JSON objects from a raw string (potentially with
        markdown and multiple JSON objects) and parses them.
        """
        # Pre-process: Remove lines that are clearly not JSON (e.g., subtask markers)
        cleaned_lines = []
        for line in raw_output.splitlines():
            if not (line.strip().startswith("--- Subtask ") and line.strip().endswith("Result ---")):
                cleaned_lines.append(line)
        processed_output = "\n".join(cleaned_lines)

        # Try multiple strategies to extract JSON
        parsed_jsons = []
        
        # Strategy 1: Try to find JSON in markdown code blocks or standalone
        json_pattern = re.compile(r"```json\s*(\{.*?\})\s*```|(\{.*?\})",
                                  re.DOTALL)
        matches = json_pattern.findall(processed_output)
        
        if matches:
            for match in matches:
                # The pattern returns tuples of (group1, group2). We need the non-empty one.
                json_str = next((group for group in match if group), None)
                if json_str:
                    try:
                        parsed_jsons.append(json.loads(json_str))
                    except json.JSONDecodeError:
                        # Try to clean and parse again
                        try:
                            # Remove extra whitespace and newlines within strings
                            cleaned_json = json_str.strip()
                            parsed_jsons.append(json.loads(cleaned_json))
                        except json.JSONDecodeError:
                            continue
        
        # Strategy 2: If no JSON found, try to parse the entire output as JSON
        if not parsed_jsons:
            try:
                parsed_jsons.append(json.loads(processed_output.strip()))
            except json.JSONDecodeError:
                pass
        
        if not parsed_jsons:
            raise json.JSONDecodeError(
                "Found potential JSON but failed to parse any valid object.",
                raw_output, 0)

        return parsed_jsons

    def _create_initial_draft_task(self, research_topic: str) -> Task:
        """Creates the task to generate the first draft of the hypothesis."""
        task_id = f"hypothesis_generation_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Precompute local RAG evidence to pass via additional_info
        rag_evidence = ""
        try:
            rag_debug = str(os.environ.get("LOCAL_RAG_DEBUG", "")).lower() in ("1", "true", "yes", "y")
            rag_evidence = run_local_rag(research_topic)
            if rag_debug:
                OutputFormatter.info(f"[RAG][READY] Generated {len(rag_evidence)} chars of RAG evidence for Subtask 1")
        except Exception as _e:
            rag_evidence = f"(RAG unavailable: {str(_e)})"
            OutputFormatter.warning(f"[RAG][ERROR] {_e}")

        # A simplified prompt focusing on the initial 4-step chain, with RAG evidence injected for Subtask 1
        content = f"""
You are a task coordinator agent. Your job is to receive a high-level goal and a precise execution plan, and then formalize this plan for a team of agents. You MUST follow the provided plan exactly without any deviation.

**MAIN GOAL**: Generate a preliminary scientific hypothesis draft for the topic: {research_topic}

**TASK DECOMPOSITION & EXECUTION PLAN**:
You MUST create and assign the following subtasks in this exact order.

**IMPORTANT DEPENDENCY FORMAT**: All task IDs follow the pattern "{task_id}.X" where X is the subtask number (1, 2, 3, etc.). Use ONLY these exact IDs in the dependencies field.

**Subtask 1: Literature Review**
- **Description**: Conduct a comprehensive literature review on '{research_topic}'. The output should be a detailed markdown report. NOTE: Additional context with RAG evidence will be provided via additional_info.
- **Assignee**: 'Scholar Scour (Literature Analysis Expert)'
- **Dependencies**: []

**Subtask 2: Creative Ideation**
- **Description**: Based on the literature review from Subtask 1, generate diverse and novel mechanisms and analogies. The output should be a markdown list of ideas.
- **Assignee**: 'Idea Igniter (Creative Innovation Specialist)'
- **Dependencies**: ["{task_id}.1"]

**Subtask 3: Technical Analysis**
- **Description**: Perform a technical analysis of the ideas from Subtask 2. The output must be a single JSON object containing a list of analyses, with scores for plausibility and consistency for each idea.
- **Assignee**: 'Dr. Qwen Technical (Technical Rigor Specialist)'
- **Dependencies**: ["{task_id}.2"]

**Subtask 4: Practical Analysis**
- **Description**: Perform a practical analysis of the ideas from Subtask 2. The output must be a single JSON object containing a list of analyses, with scores for falsifiability and feasibility for each idea.
- **Assignee**: 'Dr. Qwen Practical (Applied Research Specialist)'
- **Dependencies**: ["{task_id}.2"]

**Subtask 5: Significance Analysis**
- **Description**: Perform an impact analysis of the ideas from Subtask 2. The output must be a single JSON object containing a list of analyses, with scores for significance and broader impact for each idea.
- **Assignee**: 'Prof. Qwen Ethics (Impact & Significance Analyst)'
- **Dependencies**: ["{task_id}.2"]

**Subtask 6: Preliminary Synthesis**
- **Description**: Synthesize the literature review (Subtask 1), creative ideas (Subtask 2), and the structured JSON analyses (Subtasks 3, 4, 5) into a single, cohesive preliminary hypothesis draft. This is the final output for this task.
- **Assignee**: 'Dr. Qwen Leader (Chief Researcher & Synthesis Expert)'
- **Dependencies**: ["{task_id}.1", "{task_id}.2", "{task_id}.3", "{task_id}.4", "{task_id}.5"]

**CRITICAL INSTRUCTION FOR COORDINATOR**: Your output must be a valid plan that includes ALL the subtasks listed above, with their exact descriptions, assignees, and dependencies. The overall process will be considered complete only when Subtask 6 is finished.
"""
        # Create main task with RAG evidence stored in additional_info
        # The PROCESS_TASK_PROMPT in Workforce will inject this into subtasks
        task = Task(
            content=content.strip(),
            id=task_id,
            additional_info={
                "rag_evidence": rag_evidence,
                "instructions": """
=== LOCAL RAG EVIDENCE FOR LITERATURE REVIEW ===
The following RAG evidence is retrieved from a local knowledge graph and vector database.
Scholar Scour (Subtask 1) MUST integrate this evidence with web search and internal knowledge.

=== INTEGRATION INSTRUCTIONS FOR SCHOLAR SCOUR ===
1. Extract all references (authors, years, titles, venues) from the RAG evidence above
2. Synthesize RAG findings with your pre-trained knowledge
3. Use web search tools only for very recent developments not covered by RAG
4. Generate a comprehensive literature review with complete, properly formatted citations
5. DO NOT use placeholders in the "Key References" section

The RAG evidence is provided in the 'rag_evidence' field of this additional_info.
"""
            }
        )
        return task

    def _create_review_task(self, draft: str, research_topic: str) -> Task:
        """Creates a task to review a given draft."""
        content = f"""
**MAIN GOAL**: Review a scientific hypothesis draft and provide structured feedback.

**CONTEXT**:
- **Research Topic**: {research_topic}
- **Draft to Review**:
--- DRAFT START ---
{draft}
--- DRAFT END ---

**CRITICAL INSTRUCTION**: This is a SINGLE-STEP task that MUST be completed by ONE agent ONLY. DO NOT decompose this task into subtasks.

**TASK**:
- **Description**: Critically review the provided draft. Your output MUST be a valid JSON object containing a quality score (1-5) and actionable feedback in the format specified by Critic Crucible.
- **Assignee**: 'Critic Crucible (Peer Review Specialist)'
- **Dependencies**: []
"""
        return Task(content=content.strip())

    def _create_revision_task(self, draft: str, feedback: Dict[str, Any],
                              research_topic: str) -> Task:
        """Creates a task to revise a draft based on feedback."""
        feedback_str = json.dumps(feedback, indent=2)
        content = f"""
**MAIN GOAL**: Revise a scientific hypothesis draft based on peer review feedback.

**CONTEXT**:
- **Research Topic**: {research_topic}
- **Previous Draft**:
--- DRAFT START ---
{draft}
--- DRAFT END ---

- **Peer Review Feedback**:
--- FEEDBACK START ---
{feedback_str}
--- FEEDBACK END ---

**CRITICAL INSTRUCTION**: This is a SINGLE-STEP task that MUST be completed by Dr. Qwen Leader ONLY. DO NOT decompose this task into subtasks.

**TASK**:
- **Description**: Meticulously revise the draft to address every point in the feedback. Your goal is to improve the quality score in the next review. The output should be the new, revised markdown report.
- **Assignee**: 'Dr. Qwen Leader (Chief Researcher & Synthesis Expert)'
- **Dependencies**: []
"""
        return Task(content=content.strip())

    def _create_final_review_task(self, draft: str,
                                  research_topic: str) -> Task:
        """Creates a task for final stylistic review."""
        content = f"""
**MAIN GOAL**: Perform a final stylistic and grammatical review of a scientific hypothesis report.

**CONTEXT**:
- **Research Topic**: {research_topic}
- **Draft to Review**:
--- DRAFT START ---
{draft}
--- DRAFT END ---

**CRITICAL INSTRUCTION**: This is a SINGLE-STEP task that MUST be completed by ONE agent ONLY. DO NOT decompose this task into subtasks.

**TASK**:
- **Description**: Review the provided draft for clarity, consistency, grammar, and style. Your output MUST be a valid JSON object containing scores (1-10 scale) and actionable suggestions in the format specified by Prof. Qwen Editor.
- **Assignee**: 'Prof. Qwen Editor (Scientific Writing Specialist)'
- **Dependencies**: []
"""
        return Task(content=content.strip())

    def _create_polishing_task(self, draft: str, feedback: Dict[str, Any],
                               research_topic: str) -> Task:
        """Creates a task to polish a draft based on editorial feedback."""
        feedback_str = json.dumps(feedback, indent=2)
        content = f"""
**MAIN GOAL**: Perform a final polish on a scientific hypothesis report based on editorial feedback.

**CONTEXT**:
- **Research Topic**: {research_topic}
- **Previous Draft**:
--- DRAFT START ---
{draft}
--- DRAFT END ---

- **Editorial Feedback**:
--- FEEDBACK START ---
{feedback_str}
--- FEEDBACK END ---

**CRITICAL INSTRUCTION**: This is a SINGLE-STEP task that MUST be completed by Dr. Qwen Leader ONLY. DO NOT decompose this task into subtasks.

**TASK**:
- **Description**: Meticulously apply the editorial suggestions to improve the report's clarity, style, and professionalism. The output should be the final, polished markdown report.
- **Assignee**: 'Dr. Qwen Leader (Chief Researcher & Synthesis Expert)'
- **Dependencies**: []
"""
        return Task(content=content.strip())

    def run_research(self, research_topic: str, max_iterations: int = 3,
                     quality_threshold: float = 7.5, polish_iterations: int = 1):
        """
        Run a collaborative research session with an iterative review-revision loop.
        """
        if not self.workforce:
            self.create_research_workforce()

        self.display_agent_configs()

        OutputFormatter.header(f"Starting scientific hypothesis generation on: {research_topic}")
        print("=" * 80)

        # 1. Generate Initial Draft
        OutputFormatter.section("PHASE 1: INITIAL DRAFT GENERATION")
        try:
            initial_task = self._create_initial_draft_task(research_topic)
            OutputFormatter.info(f"Created initial draft task with ID: {initial_task.id}")
            # Debug: expected execution agents for subtasks
            OutputFormatter.info(
                "[AGENT PLAN] Subtask 1: Scholar Scour | "
                "Subtask 2: Idea Igniter | "
                "Subtask 3: Dr. Qwen Technical | "
                "Subtask 4: Dr. Qwen Practical | "
                "Subtask 5: Prof. Qwen Ethics | "
                "Subtask 6: Dr. Qwen Leader"
            )
            processed_task = self.workforce.process_task(initial_task)
            # Extract ONLY the final synthesis from Dr. Qwen Leader (last subtask result)
            raw_result = processed_task.result
            
            # DEBUG: 打印完整的原始输出
            OutputFormatter.warning("[DEBUG] Raw result length: " + str(len(raw_result)) + " chars")
            OutputFormatter.warning("[DEBUG] First 500 chars of raw result:")
            print(raw_result[:500])
            OutputFormatter.warning("[DEBUG] Last 500 chars of raw result:")
            print(raw_result[-500:])
            
            current_draft = self._extract_final_synthesis(raw_result)
            OutputFormatter.success("Initial draft generated successfully.")
            OutputFormatter.info("Draft Content Preview:")
            print(f"{current_draft[:500]}...")
        except Exception as e:
            OutputFormatter.error(f"Error during initial draft generation: {e}")
            raise

        # 2. Iterative Review and Revision Loop
        for i in range(max_iterations):
            OutputFormatter.section(f"PHASE 2: ITERATION {i + 1}/{max_iterations}")

            # a. Review the current draft
            OutputFormatter.info(f"Step 2.{i*2+1}: Reviewing draft...")
            OutputFormatter.info("[AGENT] Critic Crucible executing peer review")
            try:
                review_task = self._create_review_task(current_draft, research_topic)
                processed_review = self.workforce.process_task(review_task)

                # Parse all JSON feedback objects from the output
                all_feedback = self._extract_and_parse_json(
                    processed_review.result)

                OutputFormatter.info("--- DETAILED FEEDBACK RECEIVED ---")
                for item_idx, feedback_item in enumerate(all_feedback):
                    print(
                        f"Feedback item {item_idx + 1}: "
                        f"{json.dumps(feedback_item, indent=2)}")
                OutputFormatter.info("----------------------------------")

                # Use the last feedback item for decision making and revision
                feedback_json = all_feedback[-1]
                score = feedback_json.get("overall_quality_score", 0.0)
                OutputFormatter.success(
                    f"Review complete. Using final score for decision: "
                    f"{score}/{quality_threshold}")

            except json.JSONDecodeError as json_e:
                OutputFormatter.error(f"Failed to parse review feedback as JSON: {json_e.msg}")
                OutputFormatter.warning("Skipping this iteration due to parsing error.")
                OutputFormatter.info("--- RAW OUTPUT FROM REVIEWER ---")
                print(processed_review.result)
                OutputFormatter.info("---------------------------------")
                continue
            except Exception as e:
                OutputFormatter.error(f"Error during review task: {e}")
                continue

            # b. Decide whether to terminate or revise
            if score >= quality_threshold:
                OutputFormatter.success(f"Quality threshold met. Finalizing report.")
                break

            if i == max_iterations - 1:
                OutputFormatter.warning("Max iterations reached. Finalizing with current draft.")
                break

            # c. Revise the draft based on feedback
            OutputFormatter.info(f"Step 2.{i*2+2}: Revising draft based on feedback...")
            OutputFormatter.info("[AGENT] Dr. Qwen Leader executing revision")
            try:
                revision_task = self._create_revision_task(
                    current_draft, feedback_json, research_topic)
                processed_revision = self.workforce.process_task(revision_task)
                # Extract ONLY the revised draft from Dr. Qwen Leader
                current_draft = self._extract_final_synthesis(processed_revision.result)
                OutputFormatter.success("Revision complete.")
                OutputFormatter.info("Revised Draft Preview:")
                print(f"{current_draft}...")
            except Exception as e:
                OutputFormatter.error(f"Error during revision task: {e}")
                OutputFormatter.warning("Skipping revision and proceeding with previous draft.")
                break
        
        scientifically_sound_draft = current_draft
        OutputFormatter.section("PHASE 3: FINAL POLISHING")

        # 3. Final Polishing Loop
        polished_draft = scientifically_sound_draft
        for i in range(polish_iterations):
            OutputFormatter.info(f"Starting polishing iteration {i + 1}/{polish_iterations}...")

            # a. Get editorial feedback
            try:
                OutputFormatter.info("[AGENT] Prof. Qwen Editor executing final review")
                final_review_task = self._create_final_review_task(
                    polished_draft, research_topic)
                processed_final_review = self.workforce.process_task(
                    final_review_task)
                
                all_editorial_feedback = self._extract_and_parse_json(
                    processed_final_review.result)

                OutputFormatter.info("--- DETAILED EDITORIAL FEEDBACK ---")
                for item_idx, feedback_item in enumerate(all_editorial_feedback):
                    print(
                        f"Feedback item {item_idx + 1}: "
                        f"{json.dumps(feedback_item, indent=2)}")
                OutputFormatter.info("-----------------------------------")

                editorial_feedback = all_editorial_feedback[-1]
                OutputFormatter.success("Editorial review complete.")
                print(f"Clarity: {editorial_feedback.get('clarity_score', 'N/A')}/10, "
                      f"Consistency: {editorial_feedback.get('consistency_score', 'N/A')}/10")

            except json.JSONDecodeError as json_e:
                OutputFormatter.error(f"Failed to parse editorial feedback as JSON: {json_e.msg}")
                OutputFormatter.warning("Skipping polishing due to parsing error.")
                OutputFormatter.info("--- RAW OUTPUT FROM EDITOR ---")
                print(processed_final_review.result)
                OutputFormatter.info("---------------------------------")
                break
            except Exception as e:
                OutputFormatter.error(f"Error during final review task: {e}")
                break

            # b. Perform final polish
            try:
                OutputFormatter.info("[AGENT] Dr. Qwen Leader executing final polishing")
                polishing_task = self._create_polishing_task(
                    polished_draft, editorial_feedback, research_topic)
                processed_polish = self.workforce.process_task(polishing_task)
                # Extract ONLY the final polished output from Dr. Qwen Leader
                polished_draft = self._extract_final_synthesis(processed_polish.result)
                OutputFormatter.success("Polishing complete.")
            except Exception as e:
                    OutputFormatter.error(f"Error during polishing task: {e}")
                    break
        
        final_report_raw = polished_draft
        OutputFormatter.section("SCIENTIFIC HYPOTHESIS GENERATION COMPLETE")
        
        # Structure the final report with all intermediate processes organized
        OutputFormatter.info("Structuring final report with organized sections...")
        final_report = self._structure_final_report(final_report_raw)
        
        # Quality check on the final report
        self._perform_quality_check(final_report, research_topic)
        
        print("\n--- FINAL REPORT (PREVIEW) ---")
        print(final_report[:1000] + "\n...\n")
        print("--- END OF PREVIEW ---\n")

        # Save the research report
        try:
            saved_path = self.save_research_report(research_topic, final_report)
            OutputFormatter.success(f"Final scientific hypothesis report saved to: {saved_path}")
        except Exception as save_error:
            OutputFormatter.error(f"Could not save final report: {save_error}")

        return final_report

    def _perform_quality_check(self, result_content: str, research_topic: str):
        """Perform quality checks on the generated hypothesis report"""
        OutputFormatter.info("Performing quality assessment")
        
        # Check for content duplication
        lines = result_content.split('\n')
        duplicate_sections = []
        seen_lines = {}
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if len(line_clean) > 20:  # Only check substantial lines
                if line_clean in seen_lines:
                    duplicate_sections.append(f"Line {i+1}: '{line_clean}...'")
                else:
                    seen_lines[line_clean] = i
        
        if duplicate_sections:
            OutputFormatter.warning("Potential duplicate content detected at:")
            for item in duplicate_sections[:10]:
                print(f"  - {item}")
            if len(duplicate_sections) > 10:
                print(f"  - ... and {len(duplicate_sections) - 10} more")
        else:
            OutputFormatter.success("No obvious duplicate content detected")
 
    def _postprocess_result(self, content: str) -> str:
        """Extract final report section and remove obvious duplication/artefacts."""
        if not content:
            return content
        text = content
        # Prefer final report section if present
        final_markers = [
            "## Final Hypothesis Report",
            "# Final Hypothesis Report",
        ]
        prelim_markers = [
            "## Preliminary Hypothesis Draft",
        ]
        start_idx = -1
        for m in final_markers:
            pos = text.find(m)
            if pos != -1:
                start_idx = pos
                break
        if start_idx == -1:
            for m in prelim_markers:
                pos = text.find(m)
                if pos != -1:
                    start_idx = pos
                    break
        if start_idx != -1:
            text = text[start_idx:]
        
        # Remove repeated Subtask dump sections before the final
        lines = text.split('\n')
        cleaned_lines = []
        seen_paragraphs = set()
        buffer = []
        def flush_buffer():
            nonlocal cleaned_lines, buffer, seen_paragraphs
            if not buffer:
                return
            paragraph = '\n'.join(buffer).strip()
            key = paragraph
            if paragraph:
                if key not in seen_paragraphs:
                    cleaned_lines.extend(buffer)
                    seen_paragraphs.add(key)
            buffer = []
        
        for line in lines:
            # Drop noisy duplicated headings like multiple "### Peer Review and Refinements"
            if line.strip().startswith("--- Subtask ") and line.strip().endswith("Result ---"):
                # skip subtask markers if they appear in the final cut
                flush_buffer()
                continue
            if line.strip() == '.':
                # drop stray dots
                continue
            if line.strip() == '':
                # paragraph boundary
                flush_buffer()
                cleaned_lines.append('')
            else:
                buffer.append(line)
        flush_buffer()
        # Remove consecutive duplicate lines
        dedup_lines = []
        prev = None
        for ln in cleaned_lines:
            if prev is not None and ln.strip() and ln.strip() == prev.strip():
                continue
            dedup_lines.append(ln)
            prev = ln
        return '\n'.join(dedup_lines).strip()
 
    def save_research_report(self, research_topic: str, report_content: str) -> str:
        """Save the research report to a file with timestamp and topic information"""
        
        # Create reports directory if it doesn't exist
        reports_dir = "Scientific_Hypothesis_Reports"
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
            OutputFormatter.info(f"Created reports directory: {reports_dir}")
        
        # Generate timestamp and clean topic name for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_topic = re.sub(r'[^\w\s-]', '', research_topic)
        clean_topic = re.sub(r'\s+', '_', clean_topic.strip())
        
        # Generate filename
        filename = f"{timestamp}_{clean_topic[:50]}.md"  # Limit length to avoid filesystem issues
        filepath = os.path.join(reports_dir, filename)
        
        # Prepare the full report content with metadata
        metadata_header = f"""# Scientific Hypothesis Generation Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Research Question**: {research_topic}  
**Report ID**: {timestamp}  
**Generated by**: Scientific Hypothesis Generation Society (CAMEL + Qwen)  
**AI Research Team**: 8 Specialized Scientific Agents

---

"""
        
        full_content = metadata_header + report_content
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        return filepath


def setup_api_keys():
    """Sets up API keys for Qwen and optional Google Search."""
    # Ensuring API Keys are set for Qwen (阿里云通义千问)
    if not os.getenv("QWEN_API_KEY"):
        OutputFormatter.warning("QWEN API KEY is required to proceed. (从阿里云的百炼控制台获取)")
        qwen_api_key = "sk-2ec7d81e88334aef948d8fd42841efbe"
        os.environ["QWEN_API_KEY"] = qwen_api_key

    # 可选：如果你有自定义的 Qwen API 端点
    if not os.getenv("QWEN_API_BASE_URL"):
        OutputFormatter.info("Using default Qwen API base URL: https://dashscope.aliyuncs.com/compatible-mode/v1")
        # 如果需要自定义端点，可以取消注释下面的行
        # qwen_endpoint = input("Enter your Qwen API Base URL (按回车使用默认): ").strip()
        # if qwen_endpoint:
        #     os.environ["QWEN_API_BASE_URL"] = qwen_endpoint

    optional_keys_setup = input(
        "Setup optional API Keys for Google search functionality?(y/n): "
    ).lower()

    if "y" in optional_keys_setup:
        if not os.getenv("GOOGLE_API_KEY"):
            OutputFormatter.info("[OPTIONAL] Provide a GOOGLE CLOUD API KEY for google search")
            google_api_key = "AIzaSyBEv6w_lL9A22bqCOPT7GW4lAxsBvVeakE"
            os.environ["GOOGLE_API_KEY"] = google_api_key

        if not os.getenv("SEARCH_ENGINE_ID"):
            OutputFormatter.info("[OPTIONAL] Provide a search engine ID for google search")
            search_engine_id = "6303234664065510237"
            os.environ["SEARCH_ENGINE_ID"] = search_engine_id


def main():
    """Main function to run the Scientific Hypothesis Generation Society demo."""
    # Setup API keys first
    setup_api_keys()

    society = HypothesisGenerationSociety()

    # Example research topics across different scientific domains
    sample_topics = {
        1: {
            "topic":
            "Bridging Towers of Multi-task Learning with a Gating Mechanism for Aspect-based Sentiment Analysis and Sequential Metaphor Identification",
            "questions":
            """
            - How can a gating mechanism be designed to selectively filter and fuse information from auxiliary task towers into a main task tower, ensuring that only useful information is absorbed while irrelevant data is rejected?
            - How can the features from multiple Transformer layers within a task-specific tower be optimally combined to ensure the best use of information for downstream tasks?
            """
        },
        2: {
            "topic":
            "Bridging the Domain Gap: Improve Informal Language Translation via Counterfactual Domain Adaptation",
            "questions":
            """
            - How can counterfactual representations be generated to guide the NMT model to explore the target-domain distribution's latent space?
            - How can the generalization gap between source and target domains be bridged by constructing counterfactual interpolations?
            - How can the usefulness of source-domain samples be leveraged within a counterfactual framework to improve target-domain translation?
            """
        },
        3: {
            "topic":
            "Microbiome-Brain Communication and Neuroplasticity",
            "questions":
            """
            - What are the precise molecular mechanisms by which gut microbiota influence brain function and behavior?
            - Could microbial metabolites directly modulate synaptic plasticity and learning?
            - How might dysbiosis contribute to neurodevelopmental and neurodegenerative disorders?
            - What therapeutic interventions could leverage the microbiome-brain axis for treating neurological conditions?
            """
        }
    }

    OutputFormatter.section("SCIENTIFIC HYPOTHESIS GENERATION SOCIETY")
    print("Choose a research question or provide your own:")
    print()

    for num, info in sample_topics.items():
        print(f"{num}. {info['topic']}")
    print("4. Custom research topic")
    print()

    try:
        choice = input("Enter your choice (1-4): ").strip()

        if choice in ['1', '2', '3']:
            topic_info = sample_topics[int(choice)]
            result = society.run_research(topic_info["topic"])
        elif choice == '4':
            custom_topic = input("Enter your scientific research question: ").strip()
            result = society.run_research(custom_topic)
        else:
            OutputFormatter.warning("Invalid choice. Running default hypothesis generation")
            result = society.run_research(sample_topics[1]["topic"])

    except KeyboardInterrupt:
        OutputFormatter.info("Hypothesis generation session interrupted")
    except Exception as e:
        OutputFormatter.error(f"Error during hypothesis generation: {e}")


if __name__ == "__main__":
    main()
