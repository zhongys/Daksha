package ai

import "encoding/json"

type ReplayState struct {
	API  string          `json:"api"`
	Data json.RawMessage `json:"data"`
}
