import { RDKitModule } from '@rdkit/rdkit';
import {
  AlertCircle,
  ArrowLeft,
  CheckCircle,
  Loader2,
  MessageCircleQuestion,
  MinusCircle,
  RefreshCw,
  SkipForward,
  Sparkles,
  XCircle,
} from 'lucide-react';
import React, { useMemo, useState } from 'react';
import { MoleculeGraph, useGraphState } from './graph';
import {
  CandidateRoutePlan,
  ContextMenuState,
  Edge,
  RoutePlanContent,
  RoutePlanningResult,
  RouteProcedureStep,
  RouteStep,
  TreeNode,
} from '../types';
import { relayoutTree } from '../tree_utils';

interface RoutePlanningWorkspaceProps {
  result: RoutePlanningResult;
  activePlanId: string | null;
  onActivePlanChange: (planId: string) => void;
  rdkitModule: RDKitModule | null;
  isComputing: boolean;
  onChatAboutRoute: (planId: string, title: string) => void;
  onRefineRoute: (planId: string, title: string) => void;
  onGenerateMore: () => void;
  onSelectPlan: (route: CandidateRoutePlan) => void;
  evaluatingPlanId?: string | null;
  onSkipEvaluation: (planId: string) => void;
  selectedPlanId?: string | null;
  onReturnToSelectedPlan?: (planId: string) => void;
}

const routeTypeLabel = (routeType: string): string =>
  routeType
    .split('_')
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join(' ');

const routeStepReactionHoverInfo = (step: RouteStep): string => {
  const lines = [`# ${step.step_id}`];
  if (step.reaction_smiles) {
    lines.push('', `**Reaction SMILES:** ${step.reaction_smiles}`);
  }
  if (step.rationale) {
    lines.push('', step.rationale);
  }
  return lines.join('\n');
};

const routeStepKey = (smiles: string, canonical?: string | null): string => canonical || smiles;

const routeStepsToPreviewGraph = (
  planId: string,
  routeTitle: string,
  routeType: RoutePlanContent['route_type'],
  routeSteps: RouteStep[]
): [TreeNode[], Edge[]] => {
  const nodes: TreeNode[] = [];
  const edges: Edge[] = [];
  const nodeIdsByKey = new Map<string, string>();

  const addNode = (smiles: string, level: number, parentId: string | null): string => {
    const nodeId = `${planId}_preview_node_${nodes.length}`;
    nodes.push({
      id: nodeId,
      smiles,
      label: smiles,
      hoverInfo: `# Molecule\n\n**SMILES:** ${smiles}`,
      level,
      parentId,
      x: 0,
      y: 0,
      highlight: 'normal',
    });
    if (parentId) {
      edges.push({
        id: `${planId}_preview_edge_${edges.length}`,
        fromNode: parentId,
        toNode: nodeId,
        status: 'complete',
      });
    }
    return nodeId;
  };

  const updateSubtreeLevels = (nodeId: string, level: number): void => {
    const node = nodes.find((item) => item.id === nodeId);
    if (!node) return;
    node.level = level;
    nodes
      .filter((item) => item.parentId === nodeId)
      .forEach((child) => updateSubtreeLevels(child.id, level + 1));
  };

  const attachExistingRoot = (nodeId: string, parentId: string): boolean => {
    const node = nodes.find((item) => item.id === nodeId);
    if (!node) return false;
    if (node.parentId === parentId) return true;
    if (node.parentId !== null) return false;

    node.parentId = parentId;
    edges.push({
      id: `${planId}_preview_edge_${edges.length}`,
      fromNode: parentId,
      toNode: nodeId,
      status: 'complete',
    });
    const parent = nodes.find((item) => item.id === parentId);
    updateSubtreeLevels(nodeId, (parent?.level || 0) + 1);
    return true;
  };

  routeSteps.forEach((step) => {
    const productKey = routeStepKey(step.product_smiles, step.canonical_product_smiles);
    let productId = nodeIdsByKey.get(productKey);
    if (!productId) {
      productId = addNode(step.product_smiles, 0, null);
      nodeIdsByKey.set(productKey, productId);
    }

    const productNode = nodes.find((node) => node.id === productId);
    if (productNode) {
      productNode.reaction = {
        id: `${planId}:${step.step_id}`,
        label: 'FLASK',
        hoverInfo: `# ${routeTitle}\n\n${routeStepReactionHoverInfo(step)}`,
        highlight:
          routeType === 'template_based'
            ? 'normal'
            : routeType === 'hybrid'
              ? 'yellow'
              : 'red',
        templatesSearched: false,
      };
    }

    step.precursor_smiles.forEach((precursor, index) => {
      const precursorKey = routeStepKey(precursor, step.canonical_precursor_smiles?.[index]);
      const precursorId = nodeIdsByKey.get(precursorKey);
      if (precursorId && attachExistingRoot(precursorId, productId)) {
        return;
      }
      const newPrecursorId = addNode(precursor, (productNode?.level || 0) + 1, productId);
      if (!precursorId) {
        nodeIdsByKey.set(precursorKey, newPrecursorId);
      }
    });
  });

  return relayoutTree(nodes, edges);
};

const emptyContextMenu: ContextMenuState = {
  node: null,
  isReaction: false,
  x: 0,
  y: 0,
};

const RouteTreePreview: React.FC<{
  route: CandidateRoutePlan;
  rdkitModule: RDKitModule | null;
}> = ({ route, rdkitModule }) => {
  const graphState = useGraphState();
  const [autoZoom, setAutoZoom] = useState(true);
  const [nodes, edges] = useMemo(
    () =>
      route.route_steps && route.route_steps.length > 0
        ? routeStepsToPreviewGraph(
            route.plan_id || 'route_plan',
            route.plan.title,
            route.plan.route_type,
            route.route_steps
          )
        : [[], []],
    [route]
  );

  if (nodes.length === 0) {
    return <p className="text-sm text-tertiary">No route tree available.</p>;
  }

  return (
    <section>
      <h4 className="text-sm font-semibold text-primary mb-2">
        Retrosynthetic Route Tree Preview
      </h4>
      <div className="relative h-[360px] border border-secondary rounded overflow-hidden">
        <MoleculeGraph
          {...graphState}
          nodes={nodes}
          edges={edges}
          ctx={emptyContextMenu}
          autoZoom={autoZoom}
          setAutoZoom={setAutoZoom}
          handleNodeClick={() => {}}
          handleReactionClick={(event) => event.stopPropagation()}
          handleReactionCardClick={() => {}}
          selectedReactionNodeId={undefined}
          reactionSidebarOpen={false}
          rdkitModule={rdkitModule}
        />
      </div>
    </section>
  );
};

const DetailList: React.FC<{ title: string; items?: string[] }> = ({ title, items }) => {
  if (!items || items.length === 0) return null;
  return (
    <section>
      <h4 className="text-sm font-semibold text-primary mb-2">{title}</h4>
      <ul className="space-y-1 text-sm text-secondary">
        {items.map((item, index) => (
          <li key={`${title}-${index}`}>- {item}</li>
        ))}
      </ul>
    </section>
  );
};

const TextDetail: React.FC<{ label: string; value?: string | null }> = ({ label, value }) => {
  if (!value) return null;
  return (
    <div>
      <span className="font-medium text-primary">{label}: </span>
      <span className="text-secondary">{value}</span>
    </div>
  );
};

const formatLabel = (value: string): string =>
  value
    .split('_')
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join(' ');

const SmallBadge: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <span className="text-xs px-2 py-1 rounded bg-white/10 text-secondary">{children}</span>
);

const ConfidenceBadge: React.FC<{ confidence: RouteProcedureStep['confidence'] }> = ({
  confidence,
}) => <SmallBadge>{formatLabel(confidence)} confidence</SmallBadge>;

const EvidenceBadge: React.FC<{ evidence: RouteProcedureStep['evidence_level'] }> = ({
  evidence,
}) => <SmallBadge>{formatLabel(evidence)} evidence</SmallBadge>;

const EvaluationStatus: React.FC<{ route: CandidateRoutePlan }> = ({ route }) => {
  const evaluation = route.evaluation;
  if (!evaluation) {
    return (
      <span className="flex items-center gap-1 text-sm font-semibold text-tertiary">
        <MinusCircle className="w-4 h-4" />
        Not Evaluated
      </span>
    );
  }

  if (evaluation.accepted && !evaluation.issues?.length) {
    return (
      <span className="flex items-center gap-1 text-sm font-semibold text-success">
        <CheckCircle className="w-4 h-4" />
        Accepted
      </span>
    );
  }

  return (
    <span className="flex items-center gap-1 text-sm font-semibold text-danger">
      <XCircle className="w-4 h-4" />
      Rejected
    </span>
  );
};

const EvaluationWarnings: React.FC<{ route: CandidateRoutePlan }> = ({ route }) => {
  const warnings = route.evaluation?.warnings || [];
  if (warnings.length === 0) return null;

  return (
    <section className="alert alert-warning">
      <div className="flex items-start gap-3">
        <AlertCircle className="w-5 h-5 flex-shrink-0 text-warning" />
        <div>
          <h4 className="text-sm font-semibold text-warning">Evaluator Warnings</h4>
          <ul className="mt-2 space-y-1 text-sm text-secondary">
            {warnings.map((warning, index) => (
              <li key={`${warning.step_id || 'route'}-${index}`}>
                <span className="font-medium text-primary">
                  {formatLabel(warning.severity)}
                  {warning.step_id ? ` - ${warning.step_id}` : ''}:
                </span>{' '}
                {warning.reason}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
};

const PlanHeader: React.FC<{ route: CandidateRoutePlan; titleClassName: string }> = ({
  route,
  titleClassName,
}) => {
  const plan = route.plan;
  const evidenceLevels = Array.from(
    new Set(plan.procedure_steps.map((step) => step.evidence_level))
  );

  return (
    <div>
      <h2 className={`${titleClassName} mb-2`}>{plan.title}</h2>
      <div className="flex flex-wrap items-center gap-3 mb-2">
        <EvaluationStatus route={route} />
        <SmallBadge>{routeTypeLabel(plan.route_type)}</SmallBadge>
        {evidenceLevels.map((evidence) => (
          <EvidenceBadge key={evidence} evidence={evidence} />
        ))}
      </div>
      {plan.source_route_numbers && plan.source_route_numbers.length > 0 && (
        <div className="text-sm text-tertiary">
          Source routes: {plan.source_route_numbers.join(', ')}
        </div>
      )}
    </div>
  );
};

const PlanningNotes: React.FC<{ plan: RoutePlanContent }> = ({ plan }) => {
  const hasNotes =
    Boolean(plan.assumptions?.length) ||
    Boolean(plan.key_disconnections?.length) ||
    Boolean(plan.next_checks?.length);

  if (!hasNotes) return null;

  return (
    <details className="border-t border-secondary pt-5">
      <summary className="cursor-pointer text-sm font-semibold text-primary">
        Planning Notes
      </summary>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mt-3">
        <DetailList title="Assumptions" items={plan.assumptions} />
        <DetailList title="Key Disconnections" items={plan.key_disconnections} />
        <DetailList title="Next Checks" items={plan.next_checks} />
      </div>
    </details>
  );
};

const ProcedureStepItem: React.FC<{ step: RouteProcedureStep; index: number }> = ({
  step,
  index,
}) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <li className="border border-secondary rounded p-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2 mb-2">
            <span className="font-semibold text-primary">Step {index + 1}</span>
            <ConfidenceBadge confidence={step.confidence} />
            <EvidenceBadge evidence={step.evidence_level} />
            <SmallBadge>{step.transformation_type}</SmallBadge>
          </div>
          <p className="text-sm text-secondary leading-relaxed">{step.description}</p>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-tertiary">
            {step.source_refs.length > 0 && (
              <span>Sources: {step.source_refs.join(', ')}</span>
            )}
            <span>Yield: {step.yield_summary}</span>
          </div>
        </div>
        <button
          type="button"
          onClick={() => setExpanded((value) => !value)}
          className="btn btn-secondary btn-sm"
        >
          {expanded ? 'Hide Details' : 'Details'}
        </button>
      </div>

      {expanded && (
        <div className="mt-3 space-y-1 text-sm text-tertiary border-t border-secondary pt-3">
          <TextDetail label="Conditions" value={step.conditions_summary} />
          <TextDetail label="Evidence basis" value={step.evidence_basis} />
          <TextDetail label="Main uncertainty" value={step.main_uncertainty} />
          <div>
            <span className="font-medium text-primary">Step tree: </span>
            <span>{step.precursor_smiles.join(' + ')} -&gt; {step.product_smiles}</span>
          </div>
        </div>
      )}
    </li>
  );
};

export const SelectedRoutePlanPanel: React.FC<{
  route: CandidateRoutePlan;
  isComputing: boolean;
  onBackToPlans: (planId: string) => void;
  onChatAboutRoute: (planId: string, title: string) => void;
  onRefineRoute: (planId: string, title: string) => void;
}> = ({ route, isComputing, onBackToPlans, onChatAboutRoute, onRefineRoute }) => {
  const plan = route.plan;
  const planId = route.plan_id || '';
  const actionsDisabled = !planId || isComputing;

  return (
    <div className="card card-padding space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <PlanHeader route={route} titleClassName="text-xl font-semibold text-primary" />
        <button
          type="button"
          onClick={() => onBackToPlans(planId)}
          disabled={!planId || isComputing}
          className="btn btn-secondary"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Plans
        </button>
      </div>

      <p className="text-sm text-secondary leading-relaxed">{plan.route_strategy}</p>
      <EvaluationWarnings route={route} />
      <div className="space-y-1 text-sm">
        <TextDetail label="Route value" value={plan.route_value} />
        <TextDetail label="Evidence" value={plan.evidence_overview} />
        <TextDetail label="Novelty" value={plan.novelty_rationale} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <DetailList title="Starting Materials" items={plan.proposed_starting_materials} />
        <DetailList title="Key Risks" items={plan.key_risks} />
      </div>

      <PlanningNotes plan={plan} />

      <div className="flex flex-wrap gap-2 border-t border-secondary pt-4">
        <button
          type="button"
          onClick={() => onChatAboutRoute(planId, plan.title)}
          disabled={actionsDisabled}
          className="btn btn-secondary"
        >
          <MessageCircleQuestion className="w-4 h-4" />
          Chat About Route
        </button>
        <button
          type="button"
          onClick={() => onRefineRoute(planId, plan.title)}
          disabled={actionsDisabled}
          className="btn btn-secondary"
        >
          <Sparkles className="w-4 h-4" />
          Refine Route
        </button>
      </div>
    </div>
  );
};

const RoutePlanDetails: React.FC<{
  route: CandidateRoutePlan;
  rdkitModule: RDKitModule | null;
  isComputing: boolean;
  isEvaluating: boolean;
  onChatAboutRoute: (planId: string, title: string) => void;
  onRefineRoute: (planId: string, title: string) => void;
  onSelectPlan: (route: CandidateRoutePlan) => void;
  onSkipEvaluation: (planId: string) => void;
}> = ({
  route,
  rdkitModule,
  isComputing,
  isEvaluating,
  onChatAboutRoute,
  onRefineRoute,
  onSelectPlan,
  onSkipEvaluation,
}) => {
  const plan = route.plan;
  const planId = route.plan_id || '';
  const actionsDisabled = !planId || isComputing;
  const canSelectWithoutEvaluation = Boolean(
    route.materialized || (route.evaluation?.accepted && !route.evaluation.issues?.length)
  );

  return (
    <div className="space-y-5">
      <PlanHeader route={route} titleClassName="text-xl font-semibold text-primary" />

      <EvaluationWarnings route={route} />

      <RouteTreePreview route={route} rdkitModule={rdkitModule} />

      <section>
        <h4 className="text-sm font-semibold text-primary mb-2">Strategy</h4>
        <p className="text-sm text-secondary leading-relaxed">{plan.route_strategy}</p>
        <div className="mt-3 space-y-1 text-sm">
          <TextDetail label="Route value" value={plan.route_value} />
          <TextDetail label="Evidence" value={plan.evidence_overview} />
          <TextDetail label="Novelty" value={plan.novelty_rationale} />
        </div>
      </section>

      <section>
        <h4 className="text-sm font-semibold text-primary mb-2">Forward Procedure</h4>
        <ol className="space-y-3">
          {plan.procedure_steps.map((step, index) => (
            <ProcedureStepItem key={`${step.step_label}-${index}`} step={step} index={index} />
          ))}
        </ol>
      </section>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5 border-t border-secondary pt-5">
        <DetailList title="Starting Materials" items={plan.proposed_starting_materials} />
        <DetailList title="Key Risks" items={plan.key_risks} />
      </div>

      <PlanningNotes plan={plan} />

      {route.answer && (
        <section className="border-t border-secondary pt-5">
          <h4 className="text-sm font-semibold text-primary mb-2">Latest Answer</h4>
          <p className="text-sm text-secondary whitespace-pre-wrap">{route.answer}</p>
        </section>
      )}

      <section className="border-t border-secondary pt-5">
        {isEvaluating ? (
          <div className="alert alert-info">
            <div className="flex flex-wrap items-center justify-between gap-3 w-full">
              <div className="alert-info-text">
                <Loader2 className="w-5 h-5 animate-spin" />
                <span className="font-medium">Evaluating for potential issues...</span>
              </div>
              <button
                type="button"
                onClick={() => onSkipEvaluation(planId)}
                className="btn btn-secondary btn-sm"
              >
                <SkipForward className="w-4 h-4" />
                Skip Evaluation
              </button>
            </div>
          </div>
        ) : (
          <div className="flex flex-wrap items-center justify-between gap-2">
            <button
              type="button"
              onClick={() => onSelectPlan(route)}
              disabled={actionsDisabled}
              className="btn btn-primary"
            >
              <CheckCircle className="w-4 h-4" />
              {canSelectWithoutEvaluation ? 'Select Plan' : 'Evaluate & Select'}
            </button>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => onChatAboutRoute(planId, plan.title)}
                disabled={actionsDisabled}
                className="btn btn-secondary"
              >
                <MessageCircleQuestion className="w-4 h-4" />
                Chat About Route
              </button>
              <button
                type="button"
                onClick={() => onRefineRoute(planId, plan.title)}
                disabled={actionsDisabled}
                className="btn btn-secondary"
              >
                <Sparkles className="w-4 h-4" />
                Refine Route
              </button>
            </div>
          </div>
        )}
      </section>
    </div>
  );
};

export const RoutePlanningWorkspace: React.FC<RoutePlanningWorkspaceProps> = ({
  result,
  activePlanId,
  onActivePlanChange,
  rdkitModule,
  isComputing,
  onChatAboutRoute,
  onRefineRoute,
  onGenerateMore,
  onSelectPlan,
  evaluatingPlanId,
  onSkipEvaluation,
  selectedPlanId,
  onReturnToSelectedPlan,
}) => {
  const routes = result.candidate_routes;
  const activeRoute =
    routes.find((route) => route.plan_id === activePlanId) || routes[0] || null;

  return (
    <div className="card card-padding">
      <div className="flex flex-wrap items-start justify-between gap-4 mb-5">
        <div>
          <h2 className="text-2xl font-semibold text-primary">Candidate Route Plans</h2>
          <div className="text-sm text-tertiary mt-1">Target: {result.target_smiles}</div>
        </div>
        <div className="flex flex-wrap gap-2 mt-2">
          {selectedPlanId && onReturnToSelectedPlan && (
            <button
              type="button"
              onClick={() => onReturnToSelectedPlan(selectedPlanId)}
              disabled={isComputing}
              className="btn btn-primary"
            >
              <CheckCircle className="w-4 h-4" />
              Return to Selected Plan
            </button>
          )}
          <button
            type="button"
            onClick={onGenerateMore}
            disabled={isComputing}
            className="btn btn-secondary"
          >
            <RefreshCw className="w-4 h-4" />
            Generate More
          </button>
        </div>
      </div>

      <div className="flex flex-wrap gap-2 mb-6 border-b border-secondary pb-3">
        {routes.map((route, index) => {
          const planId = route.plan_id || `plan_${index + 1}`;
          const selected = activeRoute?.plan_id === route.plan_id;
          return (
            <button
              key={planId}
              type="button"
              onClick={() => onActivePlanChange(planId)}
              className={`btn btn-sm ${selected ? 'btn-primary' : 'btn-secondary'}`}
            >
              Plan {index + 1}
            </button>
          );
        })}
      </div>

      {activeRoute ? (
        <RoutePlanDetails
          route={activeRoute}
          rdkitModule={rdkitModule}
          isComputing={isComputing}
          isEvaluating={evaluatingPlanId === activeRoute.plan_id}
          onChatAboutRoute={onChatAboutRoute}
          onRefineRoute={onRefineRoute}
          onSelectPlan={onSelectPlan}
          onSkipEvaluation={onSkipEvaluation}
        />
      ) : (
        <p className="text-secondary">No route plans available.</p>
      )}
    </div>
  );
};
