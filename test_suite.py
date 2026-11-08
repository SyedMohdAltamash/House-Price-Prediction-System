import unittest
import json
from app import app

class HousePricePredictionTestSuite(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.valid_data = {
            "longitude": "-122.23",
            "latitude": "37.88",
            "housing_median_age": "41.0",
            "total_rooms": "880.0",
            "total_bedrooms": "129.0",
            "population": "322.0",
            "households": "126.0",
            "median_income": "8.3252",
            "ocean_proximity": "NEAR BAY"
        }

    def test_get_index(self):
        """Test home page loads with 200 OK and contains UI elements."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"House Price Prediction System", response.data)
        self.assertIn(b"Quick Samples", response.data)
        self.assertIn(b"Predict Price", response.data)

    def test_post_valid_form(self):
        """Test valid form submission returns 200 and formatted prediction."""
        response = self.client.post("/", data=self.valid_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Valuation Result", response.data)
        self.assertIn(b"$", response.data)
        # Verify entered value is retained
        self.assertIn(b"8.3252", response.data)

    def test_post_invalid_string(self):
        """Test non-numeric string gracefully shows error banner without 500 crash."""
        bad_data = self.valid_data.copy()
        bad_data["longitude"] = "invalid_text"
        response = self.client.post("/", data=bad_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Validation Error", response.data)
        self.assertIn(b"Longitude must be a valid number", response.data)

    def test_post_negative_number(self):
        """Test negative value for rooms gracefully shows validation error."""
        bad_data = self.valid_data.copy()
        bad_data["total_rooms"] = "-50"
        response = self.client.post("/", data=bad_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Validation Error", response.data)
        self.assertIn(b"cannot be negative", response.data)

    def test_post_invalid_ocean_proximity(self):
        """Test invalid ocean proximity is rejected."""
        bad_data = self.valid_data.copy()
        bad_data["ocean_proximity"] = "MARS"
        response = self.client.post("/", data=bad_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Validation Error", response.data)
        self.assertIn(b"Invalid Ocean Proximity", response.data)

    def test_api_predict_valid(self):
        """Test REST API endpoint with valid JSON."""
        payload = {
            "longitude": -122.23,
            "latitude": 37.88,
            "housing_median_age": 41.0,
            "total_rooms": 880.0,
            "total_bedrooms": 129.0,
            "population": 322.0,
            "households": 126.0,
            "median_income": 8.3252,
            "ocean_proximity": "NEAR BAY"
        }
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data.get("success"))
        self.assertIn("prediction", data)
        self.assertIn("formatted_prediction", data)
        self.assertGreater(data["prediction"], 0)

    def test_api_predict_invalid(self):
        """Test REST API endpoint handles invalid input with HTTP 400."""
        payload = {
            "longitude": "not-valid",
            "latitude": 37.88
        }
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data.get("success"))
        self.assertIn("error", data)

    def test_health_check(self):
        """Test /health endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get("status"), "ok")
        self.assertTrue(data.get("model_loaded"))

if __name__ == "__main__":
    unittest.main()
