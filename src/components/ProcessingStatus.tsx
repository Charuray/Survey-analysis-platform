import React from 'react';
import { CheckCircle, Clock, AlertCircle, Loader2 } from 'lucide-react';
import { ProcessingStep } from '../types/survey';

interface ProcessingStatusProps {
  steps: ProcessingStep[];
  currentStep?: string;
}

export const ProcessingStatus: React.FC<ProcessingStatusProps> = ({ steps, currentStep }) => {
  const getStepIcon = (step: ProcessingStep) => {
    switch (step.status) {
      case 'complete':
        return <CheckCircle className="h-5 w-5 text-green-600" />;
      case 'processing':
        return <Loader2 className="h-5 w-5 text-blue-600 animate-spin" />;
      case 'error':
        return <AlertCircle className="h-5 w-5 text-red-600" />;
      default:
        return <Clock className="h-5 w-5 text-gray-400" />;
    }
  };

  const getStepBgColor = (step: ProcessingStep) => {
    switch (step.status) {
      case 'complete':
        return 'bg-green-50 border-green-200';
      case 'processing':
        return 'bg-blue-50 border-blue-200';
      case 'error':
        return 'bg-red-50 border-red-200';
      default:
        return 'bg-gray-50 border-gray-200';
    }
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900">Processing Status</h3>
        <p className="text-sm text-gray-600 mt-1">
          Track the progress of your data analysis
        </p>
      </div>

      <div className="p-6">
        <div className="space-y-4">
          {steps.map((step, index) => (
            <div
              key={step.id}
              className={`border rounded-lg p-4 transition-all duration-200 ${getStepBgColor(step)} ${
                currentStep === step.id ? 'ring-2 ring-blue-300' : ''
              }`}
            >
              <div className="flex items-center space-x-3">
                <div className="flex-shrink-0">
                  {getStepIcon(step)}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900">
                    {step.title}
                  </p>
                  {step.message && (
                    <p className="text-xs text-gray-600 mt-1">
                      {step.message}
                    </p>
                  )}
                </div>
                <div className="flex-shrink-0">
                  <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${
                    step.status === 'complete'
                      ? 'bg-green-100 text-green-800'
                      : step.status === 'processing'
                      ? 'bg-blue-100 text-blue-800'
                      : step.status === 'error'
                      ? 'bg-red-100 text-red-800'
                      : 'bg-gray-100 text-gray-800'
                  }`}>
                    {step.status === 'complete' && 'Complete'}
                    {step.status === 'processing' && 'Processing'}
                    {step.status === 'error' && 'Error'}
                    {step.status === 'pending' && 'Pending'}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>

        {steps.length === 0 && (
          <div className="text-center py-8 text-gray-500">
            <Clock className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p>No processing steps configured</p>
            <p className="text-sm">Complete configuration to start processing</p>
          </div>
        )}
      </div>
    </div>
  );
};