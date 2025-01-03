import { NodeContent } from '../types/content';
import { aiCoreContent } from './layers/core';
import { machineLearningContent } from './layers/primary/machineLearning';
import { deepLearningContent } from './layers/primary/deepLearning';
import { computerVisionContent } from './layers/primary/computerVision';
import { neuralNetworksContent } from './layers/primary/neuralNetworks';
import { reinforcementLearningContent } from './layers/primary/reinforcementLearning';
import { aiEthicsContent } from './layers/primary/aiEthics';
import { dataScienceContent } from './layers/primary/dataScience';
import { naturalLanguageContent } from './layers/primary/naturalLanguage';

// Secondary layer imports
import { supervisedLearningContent } from './layers/secondary/supervisedLearning';
import { unsupervisedLearningContent } from './layers/secondary/unsupervisedLearning';
import { transformersContent } from './layers/secondary/transformers';
import { cnnContent } from './layers/secondary/cnn';
import { rnnContent } from './layers/secondary/rnn';
import { ganContent } from './layers/secondary/gan';
import { decisionTreesContent } from './layers/secondary/decisionTrees';
import { randomForestsContent } from './layers/secondary/randomForests';
import { svmContent } from './layers/secondary/svm';
import { clusteringContent } from './layers/secondary/clustering';
import { dimensionalityReductionContent } from './layers/secondary/dimensionalityReduction';
import { timeSeriesContent } from './layers/secondary/timeSeries';
import { recommendationSystemsContent } from './layers/secondary/recommendationSystems';
import { anomalyDetectionContent } from './layers/secondary/anomalyDetection';
import { optimizationContent } from './layers/secondary/optimization';
import { automlContent } from './layers/secondary/automl';

// Tertiary layer imports
import {
  transferLearningContent,
  featureEngineeringContent,
  modelDeploymentContent,
  hyperparameterTuningContent,
  crossValidationContent,
  ensembleMethodsContent,
  gradientDescentContent,
  backpropagationContent,
  attentionMechanismContent,
  embeddingsContent,
  fineTuningContent,
  modelCompressionContent,
  knowledgeDistillationContent,
  fewShotLearningContent,
  zeroShotLearningContent,
  activeLearningContent,
  metaLearningContent,
  neuralArchitectureContent,
  regularizationContent,
  optimizationContent as tertiaryOptimizationContent,
  lossFunctionsContent,
  metricsContent,
  evaluationContent
} from './layers/tertiary';

// Quaternary layer imports
import {
  dataAugmentationContent,
  batchNormalizationContent,
  dropoutContent,
  layerNormalizationContent,
  weightInitializationContent,
  learningRateSchedulingContent,
  earlyStoppingContent,
  checkpointingContent,
  modelSerializationContent,
  quantizationContent,
  pruningContent,
  knowledgeGraphsContent,
  adversarialTrainingContent,
  curriculumLearningContent,
  multiTaskLearningContent,
  domainAdaptationContent,
  continualLearningContent,
  onlineLearningContent,
  distributedTrainingContent,
  modelParallelismContent,
  dataParallelismContent,
  pipelineParallelismContent,
  gradientAccumulationContent,
  mixedPrecisionContent,
  modelMonitoringContent,
  abTestingContent,
  shadowDeploymentContent,
  canaryReleaseContent,
  blueGreenDeploymentContent,
  featureStoresContent,
  modelRegistryContent,
  modelVersioningContent
} from './layers/quaternary';

// Map all node content
export const nodeContentMap: Record<string, NodeContent> = {
  // Core layer
  'center': aiCoreContent,

  // Primary layer
  'primary-0': machineLearningContent,
  'primary-1': deepLearningContent,
  'primary-2': naturalLanguageContent,
  'primary-3': computerVisionContent,
  'primary-4': reinforcementLearningContent,
  'primary-5': neuralNetworksContent,
  'primary-6': dataScienceContent,
  'primary-7': aiEthicsContent,

  // Secondary layer
  'secondary-0': supervisedLearningContent,
  'secondary-1': unsupervisedLearningContent,
  'secondary-2': transformersContent,
  'secondary-3': cnnContent,
  'secondary-4': rnnContent,
  'secondary-5': ganContent,
  'secondary-6': decisionTreesContent,
  'secondary-7': randomForestsContent,
  'secondary-8': svmContent,
  'secondary-9': clusteringContent,
  'secondary-10': dimensionalityReductionContent,
  'secondary-11': timeSeriesContent,
  'secondary-12': recommendationSystemsContent,
  'secondary-13': anomalyDetectionContent,
  'secondary-14': optimizationContent,
  'secondary-15': automlContent,

  // Tertiary layer
  'tertiary-0': transferLearningContent,
  'tertiary-1': featureEngineeringContent,
  'tertiary-2': modelDeploymentContent,
  'tertiary-3': hyperparameterTuningContent,
  'tertiary-4': crossValidationContent,
  'tertiary-5': ensembleMethodsContent,
  'tertiary-6': gradientDescentContent,
  'tertiary-7': backpropagationContent,
  'tertiary-8': attentionMechanismContent,
  'tertiary-9': embeddingsContent,
  'tertiary-10': fineTuningContent,
  'tertiary-11': modelCompressionContent,
  'tertiary-12': knowledgeDistillationContent,
  'tertiary-13': fewShotLearningContent,
  'tertiary-14': zeroShotLearningContent,
  'tertiary-15': activeLearningContent,
  'tertiary-16': metaLearningContent,
  'tertiary-17': neuralArchitectureContent,
  'tertiary-18': regularizationContent,
  'tertiary-19': tertiaryOptimizationContent,
  'tertiary-20': lossFunctionsContent,
  'tertiary-21': metricsContent,
  'tertiary-22': evaluationContent,

  // Quaternary layer
  'quaternary-0': dataAugmentationContent,
  'quaternary-1': batchNormalizationContent,
  'quaternary-2': dropoutContent,
  'quaternary-3': layerNormalizationContent,
  'quaternary-4': weightInitializationContent,
  'quaternary-5': learningRateSchedulingContent,
  'quaternary-6': earlyStoppingContent,
  'quaternary-7': checkpointingContent,
  'quaternary-8': modelSerializationContent,
  'quaternary-9': quantizationContent,
  'quaternary-10': pruningContent,
  'quaternary-11': knowledgeGraphsContent,
  'quaternary-12': adversarialTrainingContent,
  'quaternary-13': curriculumLearningContent,
  'quaternary-14': multiTaskLearningContent,
  'quaternary-15': domainAdaptationContent,
  'quaternary-16': continualLearningContent,
  'quaternary-17': onlineLearningContent,
  'quaternary-18': distributedTrainingContent,
  'quaternary-19': modelParallelismContent,
  'quaternary-20': dataParallelismContent,
  'quaternary-21': pipelineParallelismContent,
  'quaternary-22': gradientAccumulationContent,
  'quaternary-23': mixedPrecisionContent,
  'quaternary-24': modelMonitoringContent,
  'quaternary-25': abTestingContent,
  'quaternary-26': shadowDeploymentContent,
  'quaternary-27': canaryReleaseContent,
  'quaternary-28': blueGreenDeploymentContent,
  'quaternary-29': featureStoresContent,
  'quaternary-30': modelRegistryContent,
  'quaternary-31': modelVersioningContent
};