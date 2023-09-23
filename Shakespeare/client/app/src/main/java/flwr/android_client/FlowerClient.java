package flwr.android_client;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.os.ConditionVariable;
import android.util.Log;
import android.util.Pair;

import androidx.lifecycle.MutableLiveData;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ExecutionException;

import org.json.JSONException;
import org.json.JSONObject;
import org.json.JSONArray;


public class FlowerClient {

    private TransferLearningModelWrapper tlModel;
    private static final int LOWER_BYTE_MASK = 0xFF;
    private MutableLiveData<Float> lastLoss = new MutableLiveData<>();
    private Context context;
    private final ConditionVariable isTraining = new ConditionVariable();
    private static String TAG = "Flower";
    private int local_epochs = 1;

    public FlowerClient(Context context, double p_val) {
        this.tlModel = new TransferLearningModelWrapper(context, p_val);
        this.context = context;
        Log.e(TAG ,  "setting model to p = " + p_val);
    }
    public void updateModel(double p_val) {
        this.tlModel = new TransferLearningModelWrapper(this.context, p_val);
        Log.e(TAG ,  "changing model to p = " + p_val);
    }

    public ByteBuffer[] getWeights() {
        return tlModel.getParameters();
    }

    public Pair<ByteBuffer[], Integer> fit(ByteBuffer[] weights, int epochs) {

        this.local_epochs = epochs;
        tlModel.updateParameters(weights);
        isTraining.close();
        tlModel.train(this.local_epochs);
        tlModel.enableTraining((epoch, loss) -> setLastLoss(epoch, loss));
        Log.e(TAG ,  "Training enabled. Local Epochs = " + this.local_epochs);
        isTraining.block();
        return Pair.create(getWeights(), tlModel.getSize_Training());
    }

    public Pair<Pair<Float, Float>, Integer> evaluate(ByteBuffer[] weights) {
        Log.e(TAG ,  "updating weights = " );
        tlModel.updateParameters(weights);
        tlModel.disableTraining();
        return Pair.create(tlModel.calculateTestStatistics(), tlModel.getSize_Testing());
    }

    public void setLastLoss(int epoch, float newLoss) {
        if (epoch == this.local_epochs - 1) {
            Log.e(TAG, "Training finished after epoch = " + epoch);
            lastLoss.postValue(newLoss);
            tlModel.disableTraining();
            isTraining.open();
        }
    }

    public void loadData(int device_id) {
        try {

            StringBuilder sb = new StringBuilder();

            String line;
            BufferedReader reader = new BufferedReader(new InputStreamReader(this.context.getAssets().open("data/" + (device_id - 1) + "_train.json")));
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
            reader.close();
            JSONObject json = new JSONObject(sb.toString());
            addSample(json, true);

            /* load test data*/
            sb = new StringBuilder();
            reader = new BufferedReader(new InputStreamReader(this.context.getAssets().open("data/" + (device_id - 1) + "_test.json")));
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
            reader.close();
            json = new JSONObject(sb.toString());
            addSample(json, false);


        } catch (IOException | JSONException ex) {
            ex.printStackTrace();
        }
    }

    private void addSample(JSONObject json, Boolean isTraining) throws IOException, JSONException {
        String characterString = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}";
        JSONArray x_list = json.getJSONArray("x");
        JSONArray y_list = json.getJSONArray("y");
        for (int i = 0; i < x_list.length(); i++) {
            Log.e(TAG, i + "th training data loaded");
            // add to the list.
            try {
                String x = x_list.getString(i);
                String y = y_list.getString(i);
                float[] indexData = new float[x.length()*80];
                Arrays.fill(indexData, 0);

                for (int j = 0; j < x.length(); j++) {
                    int classIdx = characterString.indexOf(x.charAt(j));
                    indexData[(j*80)+ classIdx] = 1;
                }
                this.tlModel.addSample(indexData, Integer.toString(characterString.indexOf(y)), isTraining).get();
            } catch (ExecutionException e) {
                throw new RuntimeException("Failed to add sample to model", e.getCause());
            } catch (InterruptedException e) {
                // no-op
            }
        }

    }
}
