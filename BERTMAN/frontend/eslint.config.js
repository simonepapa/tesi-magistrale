import { fixupConfigRules } from "@eslint/compat";
import { FlatCompat } from "@eslint/eslintrc";
import js from "@eslint/js";
import tsParser from "@typescript-eslint/parser";
import react from "eslint-plugin-react";
import reactRefresh from "eslint-plugin-react-refresh";
import globals from "globals";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const compat = new FlatCompat({
  baseDirectory: __dirname,
  recommendedConfig: js.configs.recommended,
  allConfig: js.configs.all
});

export default [
  {
    ignores: ["**/dist", "**/.eslintrc.cjs"]
  },
  ...fixupConfigRules(
    compat.extends(
      "eslint:recommended",
      "plugin:@typescript-eslint/recommended",
      "plugin:react-hooks/recommended",
      "plugin:prettier/recommended",
      "prettier"
    )
  ),
  {
    plugins: {
      "react-refresh": reactRefresh,
      react
    },
    languageOptions: {
      globals: {
        ...globals.browser
      },
      parser: tsParser
    },
    settings: {
      react: {
        version: "detect"
      }
    },
    rules: {
      "react-refresh/only-export-components": [
        "warn",
        {
          allowConstantExport: true
        }
      ],
      "react/jsx-boolean-value": ["error", "always"],
      "react/jsx-filename-extension": [
        1,
        {
          extensions: [".js", ".jsx", ".ts", ".tsx"]
        }
      ],
      "react/jsx-pascal-case": "error",
      "react/no-direct-mutation-state": "error",
      "react/prefer-stateless-function": "error",
      "react/no-multi-comp": "off",
      "react/function-component-definition": [
        "error",
        {
          namedComponents: "function-declaration",
          unnamedComponents: "function-expression"
        }
      ],
      "react/require-render-return": "error",
      "react/react-in-jsx-scope": "off",
      "prettier/prettier": "error",
      "no-console": "warn"
    }
  }
];
