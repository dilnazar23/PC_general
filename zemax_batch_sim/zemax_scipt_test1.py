import clr
import os

# Locate the ZOS-API assemblies
ZEMAX_PATH = r"C:\Program Files\Zemax OpticStudio"
clr.AddReference(os.path.join(ZEMAX_PATH, "ZOSAPI.dll"))
clr.AddReference(os.path.join(ZEMAX_PATH, "ZOSAPI_Interfaces.dll"))

from ZOSAPI import *
from ZOSAPI_Interfaces import *

zemax_app.TheSystem.LoadFile(r"C:\path\to\file.zmx", False)
lens_data = zemax_app.TheSystem.LDE
print(f"Number of surfaces: {lens_data.NumberOfSurfaces}")

# Initialize the API connection
class ZemaxApplication:
    def __init__(self):
        self.TheApplication = ZOSAPI_Connection().CreateNewApplication()
        if not self.TheApplication:
            raise Exception("Unable to connect to OpticStudio")
        self.TheSystem = self.TheApplication.PrimarySystem

    def close(self):
        self.TheApplication.CloseApplication()

# Start Zemax session
zemax_app = ZemaxApplication()
print("Connected to Zemax OpticStudio")
# Load a sample lens file
zemax_app.TheSystem.LoadFile(r"C:\path\to\your\lens_file.zmx", False)

# Access Lens Data Editor
lens_data = zemax_app.TheSystem.LDE

# Modify a parameter (e.g., radius of curvature of the first surface)
first_surface = lens_data.GetSurfaceAt(1)
first_surface.Radius = 50.0  # Set radius to 50 mm
zemax_app.TheSystem.Save()  # Save changes
# Set up ray tracing
ray_trace = zemax_app.TheSystem.Tools.OpenSequentialRayTrace()
ray_trace.ClearDetectors(0)  # Clear previous results
ray_trace.RunAndWaitForCompletion()

# Retrieve results
results = ray_trace.GetResults(0)
for ray in results:
    print(f"Ray {ray.HitObject}: X={ray.X}, Y={ray.Y}, Z={ray.Z}")
ray_trace.Close()

zemax_app.close()
print("Disconnected from Zemax OpticStudio")
