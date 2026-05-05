"use client";
import { useState } from "react";
import TextField, { type TextFieldProps } from "@mui/material/TextField";
import InputAdornment from "@mui/material/InputAdornment";
import IconButton from "@mui/material/IconButton";
import Visibility from "@mui/icons-material/Visibility";
import VisibilityOff from "@mui/icons-material/VisibilityOff";

type Props = Omit<TextFieldProps, "type" | "InputProps">;

export default function PasswordInput(props: Props) {
    const [reveal, setReveal] = useState(false);
    return (
        <TextField
            {...props}
            type={reveal ? "text" : "password"}
            slotProps={{
                input: {
                    endAdornment: (
                        <InputAdornment position="end">
                            <IconButton
                                onClick={() => setReveal((v) => !v)}
                                edge="end"
                                size="small"
                                aria-label={reveal ? "Hide password" : "Show password"}
                                aria-pressed={reveal}
                            >
                                {reveal ? <VisibilityOff fontSize="small" /> : <Visibility fontSize="small" />}
                            </IconButton>
                        </InputAdornment>
                    ),
                },
            }}
        />
    );
}
