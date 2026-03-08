declare module "react-plotly.js" {
  import { ComponentType } from "react";

  type PlotProps = {
    data: Array<Record<string, unknown>>;
    layout?: Record<string, unknown>;
    config?: Record<string, unknown>;
    style?: Record<string, unknown>;
    className?: string;
    useResizeHandler?: boolean;
  };

  const Plot: ComponentType<PlotProps>;
  export default Plot;
}
