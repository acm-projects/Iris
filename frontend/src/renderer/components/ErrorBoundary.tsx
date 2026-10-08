// Last line of defence: if a screen crashes while rendering, show a friendly
// recovery screen instead of a blank window. Details go to the console only.
import { Component, type ErrorInfo, type ReactNode } from "react";

type State = { crashed: boolean };

export default class ErrorBoundary extends Component<
  { children: ReactNode },
  State
> {
  state: State = { crashed: false };

  static getDerivedStateFromError(): State {
    return { crashed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[iris] A screen crashed:", error, info.componentStack);
  }

  render() {
    if (!this.state.crashed) return this.props.children;
    return (
      <main className="crash-screen" role="alert">
        <div>
          <h1>Something went wrong</h1>
          <p>
            Iris ran into a problem showing this screen. Your meetings and
            settings are safe. Reloading usually fixes it.
          </p>
          <button onClick={() => window.location.reload()} type="button">
            Reload Iris
          </button>
        </div>
      </main>
    );
  }
}
