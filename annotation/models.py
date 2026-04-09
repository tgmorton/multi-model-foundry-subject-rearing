"""Pydantic request/response models."""

from typing import List, Optional

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str
    token: str


class UserResponse(BaseModel):
    id: int
    username: str
    display_name: str
    role: str


class LoginResponse(BaseModel):
    token: str
    user: UserResponse


class PlacedPronounRequest(BaseModel):
    position: int = Field(ge=0)
    label: str
    lexical_form: str
    confidence: int = Field(default=3, ge=1, le=5)


class AnnotationRequest(BaseModel):
    has_null_subject: Optional[bool] = None
    pronouns: List[PlacedPronounRequest] = []
    overall_confidence: int = Field(default=3, ge=1, le=5)
    starred: bool = False
    note: str = ""
    context_expansions: int = Field(default=0, ge=0)


class ProgressResponse(BaseModel):
    total: int
    completed: int
    starred: int
    remaining: int


class AdjudicationRequest(BaseModel):
    resolution: str  # 'accept_a' | 'accept_b' | 'reannotate'
    accepted_user_id: Optional[int] = None
    has_null_subject: bool
    pronouns: List[PlacedPronounRequest] = []
    overall_confidence: int = Field(default=3, ge=1, le=5)
    note: str = ""


class AdjudicationProgressResponse(BaseModel):
    total: int
    adjudicated: int
    remaining: int
