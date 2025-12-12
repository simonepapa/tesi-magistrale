"use client";

import { Button } from "./button";
import { Calendar } from "./calendar";
import { Popover, PopoverContent, PopoverTrigger } from "./popover";
import { cn } from "@/lib/utils";
import { CalendarIcon } from "lucide-react";
import * as React from "react";

export interface DatePickerProps {
  value?: Date | null;
  onChange?: (date: Date | null) => void;
  label?: string;
  disabled?: boolean;
  disableFuture?: boolean;
  minDate?: Date;
  className?: string;
  placeholder?: string;
}

export function DatePicker({
  value,
  onChange,
  label,
  disabled = false,
  disableFuture = false,
  minDate,
  className,
  placeholder = "Pick a date"
}: DatePickerProps) {
  const [open, setOpen] = React.useState(false);

  const handleSelect = (date: Date | undefined) => {
    onChange?.(date ?? null);
    setOpen(false);
  };

  const handleClear = (e: React.MouseEvent) => {
    e.stopPropagation();
    onChange?.(null);
  };

  const disabledMatcher = React.useMemo(() => {
    const matchers = [];

    if (disableFuture) {
      matchers.push({ after: new Date() });
    }

    if (minDate) {
      matchers.push({ before: minDate });
    }

    return matchers.length > 0 ? matchers : undefined;
  }, [disableFuture, minDate]);

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      {label && (
        <label className="text-sm leading-none font-medium peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
          {label}
        </label>
      )}
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild={true}>
          <Button
            variant="outline"
            disabled={disabled}
            className={cn(
              "dark:bg-input/30 border-input focus-visible:border-ring focus-visible:ring-ring/50 h-9 w-full justify-start bg-transparent text-left font-normal shadow-xs transition-[color,box-shadow] focus-visible:ring-[3px]",
              !value && "text-muted-foreground"
            )}>
            <CalendarIcon className="mr-2 h-4 w-4" />
            {value ? (
              <span>{value.toLocaleDateString()}</span>
            ) : (
              <span>{placeholder}</span>
            )}
            {value && (
              <button
                type="button"
                onClick={handleClear}
                className="ring-offset-background focus:ring-ring ml-auto rounded-sm opacity-70 transition-opacity hover:opacity-100 focus:ring-2 focus:ring-offset-2 focus:outline-none disabled:pointer-events-none">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18"></line>
                  <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
                <span className="sr-only">Clear</span>
              </button>
            )}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="z-[9999] w-auto p-0" align="start">
          <Calendar
            mode="single"
            selected={value ?? undefined}
            onSelect={handleSelect}
            disabled={disabledMatcher}
            initialFocus={true}
          />
        </PopoverContent>
      </Popover>
    </div>
  );
}
