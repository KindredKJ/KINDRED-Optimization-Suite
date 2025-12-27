import { Agent, AgentOutputType } from './agent';
import { InputGuardrail, OutputGuardrail } from './guardrail';
import { HandoffInputFilter } from './handoff';
import { Model, ModelProvider, ModelSettings, ModelTracing } from './model';
import { RunContext } from './runContext';
import { AgentInputItem } from './types';
import { RunResult, StreamedRunResult } from './result';
import { RunHooks } from './lifecycle';
import { RunItem } from './items';
import { RunState } from './runState';
import type { Session, SessionInputCallback } from './memory/session';
/**
 * Configures settings for the entire agent run.
 */
export type RunConfig = {
    /**
     * The model to use for the entire agent run. If set, will override the model set on every
     * agent. The modelProvider passed in below must be able to resolve this model name.
     */
    model?: string | Model;
    /**
     * The model provider to use when looking up string model names. Defaults to OpenAI.
     */
    modelProvider: ModelProvider;
    /**
     * Configure global model settings. Any non-null values will override the agent-specific model
     * settings.
     */
    modelSettings?: ModelSettings;
    /**
     * A global input filter to apply to all handoffs. If `Handoff.inputFilter` is set, then that
     * will take precedence. The input filter allows you to edit the inputs that are sent to the new
     * agent. See the documentation in `Handoff.inputFilter` for more details.
     */
    handoffInputFilter?: HandoffInputFilter;
    /**
     * A list of input guardrails to run on the initial run input.
     */
    inputGuardrails?: InputGuardrail[];
    /**
     * A list of output guardrails to run on the final output of the run.
     */
    outputGuardrails?: OutputGuardrail<AgentOutputType<unknown>>[];
    /**
     * Whether tracing is disabled for the agent run. If disabled, we will not trace the agent run.
     */
    tracingDisabled: boolean;
    /**
     * Whether we include potentially sensitive data (for example: inputs/outputs of tool calls or
     * LLM generations) in traces. If false, we'll still create spans for these events, but the
     * sensitive data will not be included.
     */
    traceIncludeSensitiveData: boolean;
    /**
     * The name of the run, used for tracing. Should be a logical name for the run, like
     * "Code generation workflow" or "Customer support agent".
     */
    workflowName?: string;
    /**
     * A custom trace ID to use for tracing. If not provided, we will generate a new trace ID.
     */
    traceId?: string;
    /**
     * A grouping identifier to use for tracing, to link multiple traces from the same conversation
     * or process. For example, you might use a chat thread ID.
     */
    groupId?: string;
    /**
     * An optional dictionary of additional metadata to include with the trace.
     */
    traceMetadata?: Record<string, string>;
    /**
     * Customizes how session history is combined with the current turn's input.
     * When omitted, history items are appended before the new input.
     */
    sessionInputCallback?: SessionInputCallback;
    /**
     * Invoked immediately before calling the model, allowing callers to edit the
     * system instructions or input items that will be sent to the model.
     */
    callModelInputFilter?: CallModelInputFilter;
};
/**
 * Common run options shared between streaming and non-streaming execution pathways.
 */
type SharedRunOptions<TContext = undefined> = {
    context?: TContext | RunContext<TContext>;
    maxTurns?: number;
    signal?: AbortSignal;
    previousResponseId?: string;
    conversationId?: string;
    session?: Session;
    sessionInputCallback?: SessionInputCallback;
    callModelInputFilter?: CallModelInputFilter;
};
/**
 * Options for runs that stream incremental events as the model responds.
 */
export type StreamRunOptions<TContext = undefined> = SharedRunOptions<TContext> & {
    /**
     * Whether to stream the run. If true, the run will emit events as the model responds.
     */
    stream: true;
};
/**
 * Options for runs that collect the full model response before returning.
 */
export type NonStreamRunOptions<TContext = undefined> = SharedRunOptions<TContext> & {
    /**
     * Run to completion without streaming incremental events; leave undefined or set to `false`.
     */
    stream?: false;
};
/**
 * Options polymorphic over streaming or non-streaming execution modes.
 */
export type IndividualRunOptions<TContext = undefined> = StreamRunOptions<TContext> | NonStreamRunOptions<TContext>;
/**
 * Executes an agent workflow with the shared default `Runner` instance.
 *
 * @param agent - The entry agent to invoke.
 * @param input - A string utterance, structured input items, or a resumed `RunState`.
 * @param options - Controls streaming mode, context, session handling, and turn limits.
 * @returns A `RunResult` when `stream` is false, otherwise a `StreamedRunResult`.
 */
export declare function run<TAgent extends Agent<any, any>, TContext = undefined>(agent: TAgent, input: string | AgentInputItem[] | RunState<TContext, TAgent>, options?: NonStreamRunOptions<TContext>): Promise<RunResult<TContext, TAgent>>;
export declare function run<TAgent extends Agent<any, any>, TContext = undefined>(agent: TAgent, input: string | AgentInputItem[] | RunState<TContext, TAgent>, options?: StreamRunOptions<TContext>): Promise<StreamedRunResult<TContext, TAgent>>;
/**
 * Orchestrates agent execution, including guardrails, tool calls, session persistence, and
 * tracing. Reuse a `Runner` instance when you want consistent configuration across multiple runs.
 */
export declare class Runner extends RunHooks<any, AgentOutputType<unknown>> {
    #private;
    readonly config: RunConfig;
    /**
     * Creates a runner with optional defaults that apply to every subsequent run invocation.
     *
     * @param config - Overrides for models, guardrails, tracing, or session behavior.
     */
    constructor(config?: Partial<RunConfig>);
    /**
     * Run a workflow starting at the given agent. The agent will run in a loop until a final
     * output is generated. The loop runs like so:
     * 1. The agent is invoked with the given input.
     * 2. If there is a final output (i.e. the agent produces something of type
     *    `agent.outputType`, the loop terminates.
     * 3. If there's a handoff, we run the loop again, with the new agent.
     * 4. Else, we run tool calls (if any), and re-run the loop.
     *
     * In two cases, the agent may raise an exception:
     * 1. If the maxTurns is exceeded, a MaxTurnsExceeded exception is raised.
     * 2. If a guardrail tripwire is triggered, a GuardrailTripwireTriggered exception is raised.
     *
     * Note that only the first agent's input guardrails are run.
     *
     * @param agent - The starting agent to run.
     * @param input - The initial input to the agent. You can pass a string or an array of
     * `AgentInputItem`.
     * @param options - Options for the run, including streaming behavior, execution context, and the
     * maximum number of turns.
     * @returns The result of the run.
     */
    run<TAgent extends Agent<any, any>, TContext = undefined>(agent: TAgent, input: string | AgentInputItem[] | RunState<TContext, TAgent>, options?: NonStreamRunOptions<TContext>): Promise<RunResult<TContext, TAgent>>;
    run<TAgent extends Agent<any, any>, TContext = undefined>(agent: TAgent, input: string | AgentInputItem[] | RunState<TContext, TAgent>, options?: StreamRunOptions<TContext>): Promise<StreamedRunResult<TContext, TAgent>>;
    private readonly inputGuardrailDefs;
    private readonly outputGuardrailDefs;
}
/**
 * Mutable view of the instructions + input items that the model will receive.
 * Filters always see a copy so they can edit without side effects.
 */
export type ModelInputData = {
    input: AgentInputItem[];
    instructions?: string;
};
/**
 * Shape of the payload given to `callModelInputFilter`. Mirrored in the Python SDK so filters can
 * share the same implementation across languages.
 */
export type CallModelInputFilterArgs<TContext = unknown> = {
    modelData: ModelInputData;
    agent: Agent<TContext, AgentOutputType>;
    context: TContext | undefined;
};
/**
 * Hook invoked immediately before a model call is issued, allowing callers to adjust the
 * instructions or input array. Returning a new array enables redaction, truncation, or
 * augmentation of the payload that will be sent to the provider.
 */
export type CallModelInputFilter<TContext = unknown> = (args: CallModelInputFilterArgs<TContext>) => ModelInputData | Promise<ModelInputData>;
/**
 * Constructs the model input array for the current turn by combining the original turn input with
 * any new run items (excluding tool approval placeholders). This helps ensure that repeated calls
 * to the Responses API only send newly generated content.
 *
 * See: https://platform.openai.com/docs/guides/conversation-state?api-mode=responses.
 */
export declare function getTurnInput(originalInput: string | AgentInputItem[], generatedItems: RunItem[]): AgentInputItem[];
/**
 * Resolves the effective model for the next turn by giving precedence to the agent-specific
 * configuration when present, otherwise falling back to the runner-level default.
 */
export declare function selectModel(agentModel: string | Model, runConfigModel: string | Model | undefined): string | Model;
/**
 * Normalizes tracing configuration into the format expected by model providers.
 * Returns `false` to disable tracing, `true` to include full payload data, or
 * `'enabled_without_data'` to omit sensitive content while still emitting spans.
 */
export declare function getTracing(tracingDisabled: boolean, traceIncludeSensitiveData: boolean): ModelTracing;
export {};
