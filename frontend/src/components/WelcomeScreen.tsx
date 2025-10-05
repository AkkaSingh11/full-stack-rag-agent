import { useState } from "react";
import { InputForm } from "./InputForm";
import { AdvancedSearchSettings, SearchSettings } from "./AdvancedSearchSettings";

interface WelcomeScreenProps {
  handleSubmit: (
    submittedInputValue: string,
    effort: string,
    model: string,
    ragStrategy: string,
    hybridAlpha: number,
    ragTopK: number
  ) => void;
  onCancel: () => void;
  isLoading: boolean;
}

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({
  handleSubmit,
  onCancel,
  isLoading,
}) => {
  const [searchSettings, setSearchSettings] = useState<SearchSettings>({
    ragStrategy: "semantic",
    hybridAlpha: 0.5,
    ragTopK: 3,
  });

  const handleFormSubmit = (inputValue: string, effort: string, model: string) => {
    handleSubmit(
      inputValue,
      effort,
      model,
      searchSettings.ragStrategy,
      searchSettings.hybridAlpha,
      searchSettings.ragTopK
    );
  };

  return (
    <div className="h-full flex flex-col items-center justify-center text-center px-4 flex-1 w-full max-w-3xl mx-auto gap-4">
      <div>
        <h1 className="text-5xl md:text-6xl font-semibold text-neutral-100 mb-3">
          Welcome.
        </h1>
        <p className="text-xl md:text-2xl text-neutral-400">
          How can I help you today?
        </p>
      </div>
      <div className="w-full mt-4 space-y-3">
        <div className="flex justify-center">
          <AdvancedSearchSettings
            settings={searchSettings}
            onSettingsChange={setSearchSettings}
          />
        </div>
        <InputForm
          onSubmit={handleFormSubmit}
          isLoading={isLoading}
          onCancel={onCancel}
          hasHistory={false}
        />
      </div>
      <p className="text-xs text-neutral-500">
        Powered by Google Gemini and LangChain LangGraph.
      </p>
    </div>
  );
};
