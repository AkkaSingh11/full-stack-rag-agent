import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Settings } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

export interface SearchSettings {
  ragStrategy: string;
  hybridAlpha: number;
  ragTopK: number;
}

interface AdvancedSearchSettingsProps {
  settings: SearchSettings;
  onSettingsChange: (settings: SearchSettings) => void;
}

export function AdvancedSearchSettings({ settings, onSettingsChange }: AdvancedSearchSettingsProps) {
  const [localSettings, setLocalSettings] = useState<SearchSettings>(settings);
  const [open, setOpen] = useState(false);

  const handleApply = () => {
    onSettingsChange(localSettings);
    setOpen(false);
  };

  const strategyDescriptions = {
    semantic: "Uses AI embeddings to understand meaning and context. Best for conceptual questions.",
    keyword: "Exact keyword matching using BM25. Best for specific terms, codes, or phrases.",
    hybrid: "Combines semantic and keyword search. Best for general-purpose queries.",
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="gap-2">
          <Settings className="h-4 w-4" />
          Advanced Search Settings
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Advanced Search Settings</DialogTitle>
          <DialogDescription>
            Configure how documents are retrieved from your knowledge base
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* RAG Strategy Selection */}
          <div className="space-y-3">
            <Label>Retrieval Strategy</Label>
            <Select
              value={localSettings.ragStrategy}
              onValueChange={(value) => setLocalSettings({ ...localSettings, ragStrategy: value })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="semantic">
                  🔍 Semantic Search (AI Embeddings)
                </SelectItem>
                <SelectItem value="keyword">
                  📝 Keyword Search (BM25)
                </SelectItem>
                <SelectItem value="hybrid">
                  ⚡ Hybrid Search (Best of Both)
                </SelectItem>
              </SelectContent>
            </Select>
            <p className="text-sm text-muted-foreground">
              {strategyDescriptions[localSettings.ragStrategy as keyof typeof strategyDescriptions]}
            </p>
          </div>

          {/* Hybrid Alpha Slider (only for hybrid strategy) */}
          {localSettings.ragStrategy === "hybrid" && (
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <Label>Hybrid Balance</Label>
                <span className="text-sm text-muted-foreground">
                  {localSettings.hybridAlpha.toFixed(1)}
                </span>
              </div>
              <Slider
                value={[localSettings.hybridAlpha]}
                onValueChange={(value) => setLocalSettings({ ...localSettings, hybridAlpha: value[0] })}
                min={0}
                max={1}
                step={0.1}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>← Keyword Focus</span>
                <span>Semantic Focus →</span>
              </div>
              <p className="text-sm text-muted-foreground">
                {localSettings.hybridAlpha < 0.4
                  ? "Emphasizes exact keyword matching"
                  : localSettings.hybridAlpha > 0.6
                  ? "Emphasizes semantic understanding"
                  : "Balanced between keywords and semantics"}
              </p>
            </div>
          )}

          {/* Top-K Selection */}
          <div className="space-y-3">
            <Label>Number of Documents to Retrieve</Label>
            <Select
              value={localSettings.ragTopK.toString()}
              onValueChange={(value) => setLocalSettings({ ...localSettings, ragTopK: parseInt(value) })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1">1 document</SelectItem>
                <SelectItem value="3">3 documents</SelectItem>
                <SelectItem value="5">5 documents</SelectItem>
                <SelectItem value="10">10 documents</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-sm text-muted-foreground">
              More documents provide more context but may include less relevant information.
            </p>
          </div>
        </div>

        <div className="flex justify-end gap-2">
          <Button variant="outline" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button onClick={handleApply}>Apply Settings</Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
