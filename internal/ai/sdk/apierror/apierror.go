package apierror

// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import (
	"fmt"
	"net/http"
	"net/http/httputil"
)

// Error represents an error that originates from the API, i.e. when a request is
// made and the API returns a response with a HTTP status code. Other errors are
// not wrapped by this SDK.
type Error struct {
	Code       string `json:"code" api:"required"`
	Message    string `json:"message" api:"required"`
	Param      string `json:"param" api:"required"`
	Type       string `json:"type" api:"required"`
	StatusCode int
	Request    *http.Request
	Response   *http.Response
}

func (r *Error) Error() string {
	// Attempt to re-populate the response body
	return fmt.Sprintf("%s %q: %d %s", r.Request.Method, r.Request.URL, r.Response.StatusCode, http.StatusText(r.Response.StatusCode))
}

func (r *Error) DumpRequest(body bool) []byte {
	if r.Request.GetBody != nil {
		r.Request.Body, _ = r.Request.GetBody()
	}
	out, _ := httputil.DumpRequestOut(r.Request, body)
	return out
}

func (r *Error) DumpResponse(body bool) []byte {
	out, _ := httputil.DumpResponse(r.Response, body)
	return out
}
